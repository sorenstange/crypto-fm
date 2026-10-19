"""Offline tests for the Binance bucket client and universe classification."""

import io
import zipfile

import numpy as np
import pytest

from cfm.data import binance as bn
from cfm.data.universe import classify

NS = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'


def test_parse_listing_with_checksum_elements_and_pagination():
    # Newer entries carry ChecksumAlgorithm/ChecksumType between ETag and Size.
    xml = f"""<?xml version="1.0" encoding="UTF-8"?><ListBucketResult {NS}>
      <IsTruncated>true</IsTruncated>
      <Contents><Key>a/X-1m-2020-01.zip</Key><LastModified>2021-06-14T06:12:38.000Z</LastModified>
        <ETag>"e"</ETag><Size>10</Size><StorageClass>STANDARD</StorageClass></Contents>
      <Contents><Key>a/X-1m-2026-01.zip</Key><LastModified>2026-02-02T10:20:24.000Z</LastModified>
        <ETag>"e"</ETag><ChecksumAlgorithm>CRC64NVME</ChecksumAlgorithm>
        <ChecksumType>FULL_OBJECT</ChecksumType><Size>20</Size></Contents>
    </ListBucketResult>""".encode()
    keys, prefixes, nxt = bn.parse_listing(xml)
    assert [(k.key, k.size) for k in keys] == [("a/X-1m-2020-01.zip", 10), ("a/X-1m-2026-01.zip", 20)]
    assert prefixes == [] and nxt == "a/X-1m-2026-01.zip"


def test_parse_listing_prefixes_next_marker():
    xml = f"""<ListBucketResult {NS}><NextMarker>p/B/</NextMarker><IsTruncated>true</IsTruncated>
      <CommonPrefixes><Prefix>p/A/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>p/B/</Prefix></CommonPrefixes></ListBucketResult>""".encode()
    keys, prefixes, nxt = bn.parse_listing(xml)
    assert keys == [] and prefixes == ["p/A/", "p/B/"] and nxt == "p/B/"
    last = xml.replace(b"<IsTruncated>true", b"<IsTruncated>false")
    assert bn.parse_listing(last)[2] is None


@pytest.mark.parametrize("key,period", [
    ("data/futures/um/monthly/klines/1000PEPEUSDT/1m/1000PEPEUSDT-1m-2024-01.zip", "2024-01"),
    ("data/futures/um/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2026-10-18.zip", "2026-10-18"),
    ("data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2020-01.zip", "2020-01"),
])
def test_period_of(key, period):
    assert bn.period_of(key) == period


def test_kline_key():
    assert bn.kline_key("BTCUSDT", "2024-01") == \
        "data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    assert bn.kline_key("BTCUSDT", "2026-10-18") == \
        "data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-10-18.zip"


def test_parse_checksum():
    d = "b4e13ea185ba96c413ac9167cfbb6e238eb5eae119a7a77f41f4c61c5e099d03"
    assert bn.parse_checksum(f"{d}  BTCUSDT-1m-2020-01.zip\n".encode()) == d
    with pytest.raises(ValueError):
        bn.parse_checksum(b"<html>404</html>")


ROW_MS = b"1577836800000,7189.43,7190.52,7177,7182.44,246.092,1577836859999,1767430.16121,336,46.630,334813.19820,0\n"
HEADER = (b"open_time,open,high,low,close,volume,close_time,quote_volume,count,"
          b"taker_buy_volume,taker_buy_quote_volume,ignore\n")


@pytest.mark.parametrize("csv", [ROW_MS, HEADER + ROW_MS])
def test_read_klines_with_and_without_header(csv):
    df = bn.read_klines_csv(csv)
    assert len(df) == 1 and list(df.columns) == bn.KLINE_COLUMNS[:-1]
    assert df.loc[0, "open_time"] == 1577836800000 and df.loc[0, "low"] == 7177.0
    assert df.loc[0, "n_trades"] == 336 and df["close"].dtype == np.float64


def test_read_klines_microseconds():
    us = ROW_MS.replace(b"1577836800000", b"1577836800000000").replace(b"1577836859999", b"1577836859999000")
    df = bn.read_klines_csv(HEADER + us)
    assert df.loc[0, "open_time"] == 1577836800000 and df.loc[0, "close_time"] == 1577836859999


def test_to_epoch_ms_units():
    out = bn.to_epoch_ms(np.array([1_577_836_800, 1_577_836_800_000, 1_577_836_800_000_000]))
    assert (out == 1_577_836_800_000).all()


def test_read_funding_and_zip():
    csv = b"calc_time,funding_interval_hours,last_funding_rate\n1788220800005,8,0.00008482\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("X-fundingRate-2026-09.csv", csv)
    df = bn.read_funding_csv(bn.read_zip_csv(buf.getvalue()))
    assert df.loc[0, "calc_time"] == 1788220800005 and df.loc[0, "funding_interval_hours"] == 8


@pytest.mark.parametrize("symbol,info,excluded", [
    ("BTCUSDT", {"contractType": "PERPETUAL", "underlyingType": "COIN"}, False),
    ("1000PEPEUSDT", None, False),                 # delisted / unknown -> candidate
    ("BUSDT", None, False),                        # token "B", not a stablecoin
    ("ICPUSDT_SETTLED", None, True),
    ("AERGOUSDTSETTLEDSETTLED", None, True),
    ("BTCUSDT_250627", None, True),
    ("BTCUSDC", None, True),
    ("ETHBTC", None, True),
    ("TSLAUSDT", {"contractType": "TRADIFI_PERPETUAL", "underlyingType": "EQUITY"}, True),
    ("ALLUSDT", {"contractType": "PERPETUAL", "underlyingType": "INDEX"}, True),
    ("FOOTBALLUSDT", None, True),
    ("USDCUSDT", {"contractType": "PERPETUAL", "underlyingType": "COIN"}, True),
    ("PAXGUSDT", {"contractType": "PERPETUAL", "underlyingType": "COIN"}, True),
])
def test_classify(symbol, info, excluded):
    assert (classify(symbol, info) is not None) == excluded
