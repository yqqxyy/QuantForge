from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import sleep

import pandas as pd
import yfinance as yf
import yaml


DEFAULT_UNIVERSE = Path("data/universe_etf.yaml")
DEFAULT_OUTPUT = Path("data/prices/etf_daily.parquet")
DEFAULT_YFINANCE_CACHE = Path("data/.cache/yfinance")


@dataclass(frozen=True)
class DownloadConfig:
    symbols: list[str]
    start: str
    end: str | None
    output: Path
    retries: int
    retry_sleep: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download daily OHLCV data for a ticker universe and cache it locally."
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=DEFAULT_UNIVERSE,
        help="YAML file containing a `symbols` list.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional symbols to download. Overrides --universe when provided.",
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Start date, inclusive, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date, exclusive, in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output parquet path.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts per symbol.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between retries.",
    )
    return parser.parse_args()


def load_symbols(universe_path: Path, symbols_arg: list[str] | None) -> list[str]:
    if symbols_arg:
        symbols = symbols_arg
    else:
        if not universe_path.exists():
            raise FileNotFoundError(f"Universe file not found: {universe_path}")
        with universe_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        symbols = config.get("symbols")

    if not isinstance(symbols, list) or not symbols:
        raise ValueError("Expected a non-empty list of symbols.")

    normalized = sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
    if not normalized:
        raise ValueError("No valid symbols found.")
    return normalized


def normalize_history(symbol: str, history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    frame = history.reset_index()
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]

    if "date" not in frame.columns:
        raise ValueError(f"{symbol}: downloaded data does not contain a date column.")

    rename_map = {
        "stock_splits": "stock_splits",
        "adj_close": "adj_close",
    }
    frame = frame.rename(columns=rename_map)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.date
    frame["symbol"] = symbol

    expected_columns = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    ]
    for column in expected_columns:
        if column not in frame.columns:
            frame[column] = 0.0 if column in {"dividends", "stock_splits"} else pd.NA

    frame = frame[expected_columns].sort_values("date").reset_index(drop=True)
    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return frame


def validate_symbol_frame(symbol: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"{symbol}: no rows downloaded.")

    duplicate_dates = frame["date"].duplicated().sum()
    if duplicate_dates:
        raise ValueError(f"{symbol}: found {duplicate_dates} duplicate dates.")

    price_columns = ["open", "high", "low", "close", "adj_close"]
    invalid_prices = (frame[price_columns] <= 0).any(axis=1).sum()
    if invalid_prices:
        raise ValueError(f"{symbol}: found {invalid_prices} rows with non-positive prices.")

    missing_prices = frame[price_columns].isna().any(axis=1).sum()
    if missing_prices:
        raise ValueError(f"{symbol}: found {missing_prices} rows with missing prices.")

    invalid_volume = (frame["volume"] < 0).sum()
    if invalid_volume:
        raise ValueError(f"{symbol}: found {invalid_volume} rows with negative volume.")


def download_symbol(symbol: str, start: str, end: str | None, retries: int, retry_sleep: float) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                actions=True,
                repair=True,
            )
            frame = normalize_history(symbol, history)
            validate_symbol_frame(symbol, frame)
            logging.info("%-6s rows=%5d first=%s last=%s", symbol, len(frame), frame["date"].min(), frame["date"].max())
            return frame
        except Exception as exc:
            last_error = exc
            logging.warning("%s attempt %d/%d failed: %s", symbol, attempt, retries, exc)
            if attempt < retries:
                sleep(retry_sleep)

    raise RuntimeError(f"{symbol}: failed after {retries} attempts") from last_error


def build_config(args: argparse.Namespace) -> DownloadConfig:
    symbols = load_symbols(args.universe, args.symbols)
    end = args.end or date.today().isoformat()
    return DownloadConfig(
        symbols=symbols,
        start=args.start,
        end=end,
        output=args.output,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )


def save_prices(frames: list[pd.DataFrame], output: Path) -> pd.DataFrame:
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(output, index=False)
    return prices


def configure_yfinance_cache() -> None:
    DEFAULT_YFINANCE_CACHE.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(DEFAULT_YFINANCE_CACHE))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    try:
        configure_yfinance_cache()
        config = build_config(args)
        logging.info("Downloading %d symbols from %s to %s", len(config.symbols), config.start, config.end)

        frames = [
            download_symbol(
                symbol=symbol,
                start=config.start,
                end=config.end,
                retries=config.retries,
                retry_sleep=config.retry_sleep,
            )
            for symbol in config.symbols
        ]
        prices = save_prices(frames, config.output)

        logging.info("Saved %d rows to %s", len(prices), config.output)
        logging.info("Symbols: %s", ", ".join(sorted(prices["symbol"].unique())))
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
