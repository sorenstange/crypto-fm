"""GPU smoke test (M-1): device info, tiny bf16 forward/backward, SDPA flash kernel,
torch.compile. Exits non-zero if any check fails.

    python -m cfm.smoke.gpu_check
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class TinyBlock(nn.Module):
    """Pre-norm causal attention + MLP, the same ops the backbone will use."""

    def __init__(self, d: int = 128, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm2 = nn.RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.n_heads, d // self.n_heads).unbind(2)
        att = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + self.proj(att.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.norm2(x))


class TinyModel(nn.Module):
    def __init__(self, d_in: int = 32, d: int = 128, n_layers: int = 2, n_out: int = 66):
        super().__init__()
        self.embed = nn.Linear(d_in, d)
        self.blocks = nn.ModuleList(TinyBlock(d) for _ in range(n_layers))
        self.head = nn.Linear(d, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


def fwd_bwd(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, torch.dtype]:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    return loss.item(), gnorm.item(), logits.dtype


def main() -> int:
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= cond
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)

    print(f"torch {torch.__version__}, CUDA runtime {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
    check("cuda available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name}, {props.total_memory / 2**30:.1f} GiB, sm_{props.major}{props.minor}, "
          f"{props.multi_processor_count} SMs")
    check("bf16 supported", torch.cuda.is_bf16_supported())

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    dev = torch.device("cuda")
    model = TinyModel().to(dev)
    x = torch.randn(16, 128, 32, device=dev)
    y = torch.randint(0, 66, (16, 128), device=dev)

    # 1) eager bf16 forward/backward; initial CE should be near ln(66) = 4.19
    loss, gnorm, dtype = fwd_bwd(model, x, y)
    check("bf16 autocast output dtype", dtype == torch.bfloat16, str(dtype))
    check("loss finite and ~ln(66)", abs(loss - 4.19) < 0.5, f"loss={loss:.4f}")
    check("grads finite", gnorm == gnorm and gnorm < float("inf"), f"grad_norm={gnorm:.4f}")

    # 2) a few optimizer steps reduce the loss
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(20):
        fwd_bwd(model, x, y)
        opt.step()
    loss_after, _, _ = fwd_bwd(model, x, y)
    check("training reduces loss", loss_after < loss, f"{loss:.4f} -> {loss_after:.4f}")

    # 3) flash attention kernel available for bf16 (fwd + bwd)
    try:
        q = torch.randn(4, 4, 128, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(q, q, q, is_causal=True).sum().backward()
        check("SDPA flash attention bf16 fwd/bwd", True)
    except RuntimeError as e:
        check("SDPA flash attention bf16 fwd/bwd", False, str(e).splitlines()[0])

    # 4) torch.compile matches eager
    try:
        torch.manual_seed(0)
        eager = TinyModel().to(dev)
        twin = TinyModel().to(dev)
        twin.load_state_dict(eager.state_dict())
        compiled = torch.compile(twin)
        t0 = time.time()
        l_c, _, _ = fwd_bwd(compiled, x, y)
        t_compile = time.time() - t0
        l_e, _, _ = fwd_bwd(eager, x, y)
        check("torch.compile fwd/bwd matches eager", abs(l_c - l_e) < 2e-2,
              f"eager={l_e:.4f} compiled={l_c:.4f} (compile {t_compile:.0f}s)")
    except Exception as e:  # noqa: BLE001 -- report any compile failure
        check("torch.compile fwd/bwd", False, f"{type(e).__name__}: {str(e).splitlines()[0]}")

    # 5) rough throughput of the tiny model
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        fwd_bwd(model, x, y)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 50
    print(f"tiny model fwd+bwd: {dt * 1e3:.1f} ms/iter, peak mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")

    print("GPU CHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
