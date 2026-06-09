from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml
import yfinance as yf


THEME_DIR = Path("data/themes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a theme YAML config for run_theme_trends.py.")
    parser.add_argument("--name", required=True, help='Theme display name, e.g. "Uranium And Nuclear Theme".')
    parser.add_argument("--symbols", nargs="+", required=True, help="Ticker symbols in the theme basket.")
    parser.add_argument("--slug", default=None, help="Optional slug. Defaults to a normalized version of --name.")
    parser.add_argument("--description", default=None, help="Optional theme description.")
    parser.add_argument("--benchmark", default=None, help="Optional benchmark ticker. Defaults to the first symbol.")
    parser.add_argument("--leader", default=None, help="Optional leader ticker. Defaults to benchmark.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output YAML path.")
    parser.add_argument(
        "--no-fetch-labels",
        action="store_true",
        help="Do not query Yahoo Finance metadata; use ticker symbols as labels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file when it already exists.",
    )
    return parser.parse_args()


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    slug = re.sub(r"_(theme|monitor|watchlist|basket)$", "", slug)
    if not slug:
        raise ValueError("Could not infer a valid slug from --name.")
    return slug


def normalize_symbols(symbols: list[str]) -> list[str]:
    normalized = []
    seen = set()
    for symbol in symbols:
        value = symbol.strip().upper()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    if not normalized:
        raise ValueError("Expected at least one valid ticker symbol.")
    return normalized


def fetch_label(symbol: str) -> str:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.get_info()
        for key in ["longName", "shortName", "displayName", "quoteType"]:
            value = info.get(key)
            if value:
                return str(value)
    except Exception as exc:
        print(f"warning: could not fetch label for {symbol}: {exc}", file=sys.stderr)
    return symbol


def build_labels(symbols: list[str], fetch_labels: bool) -> dict[str, str]:
    if not fetch_labels:
        return {symbol: symbol for symbol in symbols}
    return {symbol: fetch_label(symbol) for symbol in symbols}


def yaml_text(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=False, width=1000)


def main() -> int:
    try:
        args = parse_args()
        symbols = normalize_symbols(args.symbols)
        slug = args.slug or slugify(args.name)
        benchmark = args.benchmark.upper() if args.benchmark else symbols[0]
        leader = args.leader.upper() if args.leader else benchmark
        output = args.output or THEME_DIR / f"{slug}.yaml"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")

        config = {
            "name": args.name,
            "slug": slug,
            "description": args.description or f"{args.name} trend monitor.",
            "benchmark": benchmark,
            "leader": leader,
            "default_prices": f"data/prices/{slug}_daily.parquet",
            "default_report_dir": f"reports/themes/{slug}",
            "symbols": symbols,
            "labels": build_labels(symbols, fetch_labels=not args.no_fetch_labels),
        }

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml_text(config), encoding="utf-8")
        print(f"Wrote {output}")
        print("\nNext commands:")
        print(
            "conda run -n quantforge python download_prices.py "
            f"--universe {output} --start 2024-01-01 --output data/prices/{slug}_daily.parquet"
        )
        print(f"conda run -n quantforge python run_theme_trends.py --theme {output}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
