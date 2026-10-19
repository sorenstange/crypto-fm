"""Client for the Binance public data bucket (data.binance.vision), USD-M futures.

Shared by universe selection, the downloader and the converter:
- bucket listing (S3 XML, paginated) via `list_prefixes` / `list_keys`
- HTTP GETs with retries and exponential backoff, one session per thread
- `.CHECKSUM` parsing and SHA-256 verification
- kline / fundingRate CSV parsing; the header row and the timestamp unit vary between
  files, so both are detected per file.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DATA_URL = "https://data.binance.vision"
UM = "data/futures/um"
EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"

_S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def data_root() -> Path:
    """DATA_ROOT from jobs/common.sh, defaulting to <project>/data."""
    return Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[3] / "data"))


# --- HTTP -----------------------------------------------------------------------------

_local = threading.local()


def session() -> requests.Session:
    """A per-thread session retrying connection errors, 429 and 5xx with backoff."""
    s = getattr(_local, "session", None)
    if s is None:
        retry = Retry(total=6, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET", "HEAD"), respect_retry_after_header=True)
        s = requests.Session()
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=4))
        _local.session = s
    return s


def get(url: str, timeout: float = 60.0) -> bytes | None:
    """GET `url`; None on 404, raises on any other failure after retries."""
    r = session().get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


# --- bucket listing -------------------------------------------------------------------

@dataclass(frozen=True)
class Key:
    key: str
    size: int
    last_modified: str

    @property
    def url(self) -> str:
        return f"{DATA_URL}/{self.key}"


def parse_listing(xml: bytes) -> tuple[list[Key], list[str], str | None]:
    """One S3 ListBucketResult page -> (keys, common prefixes, marker for the next page).

    Parses the XML properly: newer <Contents> entries carry extra elements
    (ChecksumAlgorithm, ChecksumType) that break naive regex parsing.
    """
    root = ET.fromstring(xml)
    keys = [Key(c.findtext("s:Key", namespaces=_S3_NS),
                int(c.findtext("s:Size", namespaces=_S3_NS)),
                c.findtext("s:LastModified", namespaces=_S3_NS))
            for c in root.findall("s:Contents", _S3_NS)]
    prefixes = [p.findtext("s:Prefix", namespaces=_S3_NS)
                for p in root.findall("s:CommonPrefixes", _S3_NS)]
    nxt = None
    if root.findtext("s:IsTruncated", namespaces=_S3_NS) == "true":
        # NextMarker is only returned with a delimiter; otherwise continue after the last key.
        nxt = root.findtext("s:NextMarker", namespaces=_S3_NS) or max(
            [k.key for k in keys] + prefixes)
    return keys, prefixes, nxt


def _list(prefix: str, delimiter: str | None) -> Iterator[tuple[list[Key], list[str]]]:
    marker = ""
    while True:
        params = {"prefix": prefix, "marker": marker}
        if delimiter:
            params["delimiter"] = delimiter
        r = session().get(LIST_URL, params=params, timeout=60)
        r.raise_for_status()
        keys, prefixes, marker = parse_listing(r.content)
        yield keys, prefixes
        if marker is None:
            return


def list_prefixes(prefix: str) -> list[str]:
    """Immediate sub-'directories' of `prefix` (which must end with '/'), names only."""
    out = []
    for _, prefixes in _list(prefix, "/"):
        out += [p[len(prefix):].rstrip("/") for p in prefixes if p != prefix]
    return out


def list_keys(prefix: str) -> list[Key]:
    """All objects under `prefix` (recursive)."""
    return [k for keys, _ in _list(prefix, None) for k in keys]


# --- URLs -----------------------------------------------------------------------------

def kline_prefix(symbol: str, interval: str = "1m", freq: str = "monthly",
                 kind: str = "klines") -> str:
    return f"{UM}/{freq}/{kind}/{symbol}/{interval}/"


def kline_key(symbol: str, period: str, interval: str = "1m", kind: str = "klines") -> str:
    """period 'YYYY-MM' (monthly file) or 'YYYY-MM-DD' (daily file)."""
    freq = "daily" if len(period) == 10 else "monthly"
    return f"{kline_prefix(symbol, interval, freq, kind)}{symbol}-{interval}-{period}.zip"


def funding_key(symbol: str, period: str) -> str:
    return f"{UM}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{period}.zip"


def period_of(key: str) -> str:
    """'.../BTCUSDT-1m-2024-01.zip' -> '2024-01'; daily files give 'YYYY-MM-DD'."""
    parts = key.rsplit("/", 1)[-1].removesuffix(".zip").split("-")
    # Trailing all-digit parts are the date: [YYYY, MM] or [YYYY, MM, DD].
    date = []
    for p in reversed(parts):
        if not p.isdigit():
            break
        date.append(p)
    return "-".join(reversed(date))


# --- checksums ------------------------------------------------------------------------

def parse_checksum(text: bytes | str) -> str:
    """'<sha256>  <filename>' -> sha256 hex digest (lower case)."""
    if isinstance(text, bytes):
        text = text.decode()
    digest = text.split()[0].lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"malformed CHECKSUM: {text[:100]!r}")
    return digest


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- CSV parsing ----------------------------------------------------------------------

def read_zip_csv(data: bytes) -> bytes:
    """The single CSV inside a Binance zip (also checks the zip CRC)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in zip, got {z.namelist()}")
        return z.read(names[0])


def has_header(csv: bytes) -> bool:
    """Some files start with a header row, some don't."""
    first = csv.split(b"\n", 1)[0].split(b",", 1)[0].strip()
    try:
        float(first)
        return False
    except ValueError:
        return True


def to_epoch_ms(ts: np.ndarray) -> np.ndarray:
    """Epoch timestamps in s, ms or µs (detected per value by magnitude) -> int64 ms.

    Binance switched spot files to µs from 2025; futures files checked so far are ms.
    """
    ts = np.asarray(ts, dtype=np.int64)
    out = ts.copy()
    us = ts >= 10**14            # µs: ~1.6e15 for 2020+
    sec = ts < 10**11            # s:  ~1.6e9
    out[us] //= 1000
    out[sec] *= 1000
    return out


def read_klines_csv(csv: bytes) -> pd.DataFrame:
    """Kline CSV -> DataFrame with KLINE_COLUMNS (minus `ignore`); times as int64 ms."""
    df = pd.read_csv(io.BytesIO(csv), header=None, names=KLINE_COLUMNS,
                     skiprows=1 if has_header(csv) else 0, usecols=KLINE_COLUMNS[:-1],
                     dtype={c: "float64" for c in KLINE_COLUMNS[1:6] + KLINE_COLUMNS[7:11]}
                     | {"open_time": "int64", "close_time": "int64", "n_trades": "int64"})
    df["open_time"] = to_epoch_ms(df["open_time"].to_numpy())
    df["close_time"] = to_epoch_ms(df["close_time"].to_numpy())
    return df


def read_funding_csv(csv: bytes) -> pd.DataFrame:
    """fundingRate CSV -> calc_time (int64 ms, raw; has ~0-20 ms jitter), interval h, rate."""
    df = pd.read_csv(io.BytesIO(csv), header=None, names=FUNDING_COLUMNS,
                     skiprows=1 if has_header(csv) else 0,
                     dtype={"calc_time": "int64", "funding_interval_hours": "int64",
                            "last_funding_rate": "float64"})
    df["calc_time"] = to_epoch_ms(df["calc_time"].to_numpy())
    return df
