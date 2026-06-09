from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import yfinance as yf

from create_theme_config import slugify


THEME_DIR = Path("data/themes")
DISCOVERY_DIR = Path("reports/theme_discovery")
DEFAULT_QUOTE_TYPES = {"EQUITY", "ETF", "FUTURE"}
PREFERRED_EXCHANGES = {
    "ASE",
    "BATS",
    "BTS",
    "CBOE",
    "CME",
    "CMX",
    "NYB",
    "NYM",
    "NYQ",
    "NMS",
    "NASDAQ",
    "PCX",
}

BUILT_IN_THEME_HINTS = [
    (
        {"ai", "chip", "chips", "semiconductor", "semiconductors", "gpu", "accelerator"},
        ["SMH", "SOXX", "NVDA", "AMD", "AVGO", "MRVL", "MU", "TSM", "ASML", "AMAT", "LRCX", "KLAC", "ARM", "QCOM"],
    ),
    (
        {"oil", "crude", "brent", "wti"},
        ["USO", "CL=F", "BZ=F", "BNO", "XLE", "XOM", "CVX", "OXY", "COP", "SLB", "HAL", "OIH", "NG=F", "RB=F"],
    ),
    (
        {"uranium", "nuclear", "reactor", "reactors"},
        ["URA", "URNM", "CCJ", "UEC", "NXE", "DNN", "SMR", "CEG", "BWXT", "NNE", "LEU"],
    ),
    (
        {"solar", "renewable", "renewables", "clean", "energy"},
        ["TAN", "ICLN", "FSLR", "ENPH", "SEDG", "NXT", "RUN", "BEPC", "NEE"],
    ),
    (
        {"battery", "lithium", "ev", "electric", "vehicle"},
        ["LIT", "BATT", "TSLA", "ALB", "SQM", "LAC", "RIVN", "ON", "MP"],
    ),
    (
        {"cybersecurity", "cyber", "security"},
        ["CIBR", "HACK", "PANW", "CRWD", "ZS", "FTNT", "OKTA", "S", "NET"],
    ),
]


@dataclass
class Candidate:
    symbol: str
    quote_type: str
    exchange: str
    name: str
    source_query: str
    rank: int
    raw_score: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover Yahoo Finance tickers for a theme and write a theme YAML.")
    parser.add_argument("--theme-name", required=True, help='Theme name, e.g. "AI chip" or "oil geopolitics".')
    parser.add_argument("--slug", default=None, help="Optional slug. Defaults to a normalized version of --theme-name.")
    parser.add_argument("--description", default=None, help="Optional theme description.")
    parser.add_argument("--max-symbols", type=int, default=12, help="Maximum selected symbols.")
    parser.add_argument("--max-results-per-query", type=int, default=20, help="Yahoo results per search query.")
    parser.add_argument("--queries", nargs="*", default=None, help="Optional extra search queries.")
    parser.add_argument("--seed-symbols", nargs="*", default=None, help="Optional tickers to force-include and boost.")
    parser.add_argument("--no-built-in-hints", action="store_true", help="Disable built-in theme seed ticker hints.")
    parser.add_argument("--benchmark", default=None, help="Optional benchmark ticker. Defaults to best ETF/future/first result.")
    parser.add_argument("--leader", default=None, help="Optional leader ticker. Defaults to best equity/benchmark.")
    parser.add_argument(
        "--quote-types",
        nargs="*",
        default=sorted(DEFAULT_QUOTE_TYPES),
        help="Allowed quote types, e.g. EQUITY ETF FUTURE INDEX.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional YAML output path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    parser.add_argument("--skip-validation", action="store_true", help="Do not verify candidates have recent price history.")
    return parser.parse_args()


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def tokenize(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(token) >= 2}


def build_queries(theme_name: str, extra_queries: list[str] | None) -> list[str]:
    base = theme_name.strip()
    queries = [
        base,
        f"{base} ETF",
        f"{base} stock",
        f"{base} stocks",
        f"{base} companies",
    ]
    lowered = base.lower()
    if any(word in lowered for word in ["oil", "crude", "gas", "commodity", "gold", "copper", "wheat"]):
        queries.extend([f"{base} futures", f"{base} commodity", f"{base} producer"])
    if any(word in lowered for word in ["ai", "chip", "semiconductor", "nuclear", "uranium", "battery", "solar"]):
        queries.extend([f"{base} technology", f"{base} industry"])
    if extra_queries:
        queries.extend(extra_queries)

    seen = set()
    output = []
    for query in queries:
        normalized = " ".join(query.split())
        if normalized and normalized.lower() not in seen:
            output.append(normalized)
            seen.add(normalized.lower())
    return output


def built_in_seed_symbols(theme_name: str) -> list[str]:
    theme_tokens = tokenize(theme_name)
    output: list[str] = []
    seen = set()
    for keywords, symbols in BUILT_IN_THEME_HINTS:
        if theme_tokens & keywords:
            for symbol in symbols:
                if symbol not in seen:
                    output.append(symbol)
                    seen.add(symbol)
    return output


def candidate_name(quote: dict[str, Any]) -> str:
    for key in ["longname", "shortname", "displayName"]:
        value = quote.get(key)
        if value:
            return str(value)
    return str(quote.get("symbol", ""))


def score_quote(quote: dict[str, Any], query: str, rank: int, theme_tokens: set[str], seed_symbols: set[str]) -> float:
    symbol = normalize_symbol(str(quote.get("symbol", "")))
    quote_type = str(quote.get("quoteType", "")).upper()
    exchange = str(quote.get("exchange", "")).upper()
    name = candidate_name(quote)
    raw = float(quote.get("score") or 0.0)

    score = 100.0 - rank * 4.0
    score += min(raw / 1000.0, 25.0)
    if quote_type == "ETF":
        score += 12.0
    elif quote_type == "EQUITY":
        score += 8.0
    elif quote_type == "FUTURE":
        score += 6.0
    elif quote_type == "INDEX":
        score -= 8.0
    if exchange in PREFERRED_EXCHANGES:
        score += 8.0
    if "." in symbol and not symbol.endswith("=F"):
        score -= 10.0
    if symbol in seed_symbols:
        score += 80.0

    name_tokens = tokenize(f"{symbol} {name} {query}")
    score += 6.0 * len(theme_tokens & name_tokens)
    return score


def search_candidates(
    queries: list[str],
    max_results_per_query: int,
    allowed_quote_types: set[str],
    seed_symbols: list[str],
) -> list[Candidate]:
    theme_tokens = tokenize(" ".join(queries))
    seed_set = set(seed_symbols)
    seed_rank = {symbol: index for index, symbol in enumerate(seed_symbols)}
    candidates: dict[str, Candidate] = {}
    for query in queries:
        try:
            search = yf.Search(query, max_results=max_results_per_query, news_count=0, lists_count=0, recommended=0)
            quotes = search.quotes or []
        except Exception as exc:
            print(f"warning: search failed for {query!r}: {exc}", file=sys.stderr)
            continue
        for rank, quote in enumerate(quotes, start=1):
            symbol = normalize_symbol(str(quote.get("symbol", "")))
            quote_type = str(quote.get("quoteType", "")).upper()
            if not symbol or quote_type not in allowed_quote_types:
                continue
            candidate = Candidate(
                symbol=symbol,
                quote_type=quote_type,
                exchange=str(quote.get("exchange", "")).upper(),
                name=candidate_name(quote),
                source_query=query,
                rank=rank,
                raw_score=float(quote.get("score") or 0.0),
                score=score_quote(quote, query, rank, theme_tokens, seed_set),
            )
            if symbol in seed_rank:
                candidate.score += max(0.0, 40.0 - seed_rank[symbol] * 1.5)
            previous = candidates.get(symbol)
            if previous is None or candidate.score > previous.score:
                candidates[symbol] = candidate

    for symbol in seed_symbols:
        if symbol not in candidates:
            candidates[symbol] = Candidate(
                symbol=symbol,
                quote_type="SEED",
                exchange="",
                name=fetch_label(symbol),
                source_query="seed",
                rank=0,
                raw_score=0.0,
                score=180.0 - seed_rank[symbol] * 2.0,
            )
    return sorted(candidates.values(), key=lambda item: item.score, reverse=True)


def fetch_label(symbol: str) -> str:
    try:
        info = yf.Ticker(symbol).get_info()
        for key in ["longName", "shortName", "displayName", "quoteType"]:
            value = info.get(key)
            if value:
                return str(value)
    except Exception:
        pass
    return symbol


def has_recent_history(symbol: str) -> bool:
    try:
        history = yf.Ticker(symbol).history(period="2mo", interval="1d", auto_adjust=False)
        return not history.empty
    except Exception:
        return False


def validate_candidates(candidates: list[Candidate], skip_validation: bool) -> list[Candidate]:
    if skip_validation:
        return candidates
    output = []
    for candidate in candidates:
        if has_recent_history(candidate.symbol):
            output.append(candidate)
        else:
            print(f"warning: dropping {candidate.symbol}; no recent price history found", file=sys.stderr)
    return output


def choose_benchmark(candidates: list[Candidate], explicit: str | None) -> str:
    if explicit:
        return normalize_symbol(explicit)
    for quote_type in ["ETF", "FUTURE", "EQUITY", "SEED"]:
        for candidate in candidates:
            if candidate.quote_type == quote_type:
                return candidate.symbol
    raise ValueError("No candidates available to choose benchmark.")


def is_fund_or_future(candidate: Candidate) -> bool:
    name = candidate.name.lower()
    return (
        candidate.quote_type in {"ETF", "FUTURE", "INDEX"}
        or candidate.symbol.endswith("=F")
        or "etf" in name
        or "fund" in name
        or "index" in name
        or "futures" in name
    )


def choose_leader(candidates: list[Candidate], benchmark: str, explicit: str | None) -> str:
    if explicit:
        return normalize_symbol(explicit)
    for candidate in candidates:
        if candidate.symbol != benchmark and not is_fund_or_future(candidate):
            return candidate.symbol
    for quote_type in ["EQUITY", "FUTURE", "ETF", "SEED"]:
        for candidate in candidates:
            if candidate.quote_type == quote_type and candidate.symbol != benchmark:
                return candidate.symbol
    return benchmark


def write_outputs(
    args: argparse.Namespace,
    candidates: list[Candidate],
    selected: list[Candidate],
    benchmark: str,
    leader: str,
    slug: str,
    output: Path,
    queries: list[str],
) -> None:
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")

    config = {
        "name": args.theme_name,
        "slug": slug,
        "description": args.description or f"{args.theme_name} trend monitor.",
        "benchmark": benchmark,
        "leader": leader,
        "default_prices": f"data/prices/{slug}_daily.parquet",
        "default_report_dir": f"reports/themes/{slug}",
        "symbols": [candidate.symbol for candidate in selected],
        "labels": {candidate.symbol: candidate.name for candidate in selected},
        "discovery": {
            "method": "yfinance_search",
            "queries": queries,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False, width=1000), encoding="utf-8")

    discovery_dir = DISCOVERY_DIR / slug
    discovery_dir.mkdir(parents=True, exist_ok=True)
    rows = [candidate.__dict__ for candidate in candidates]
    pd.DataFrame(rows).to_csv(discovery_dir / "candidates.csv", index=False)
    pd.DataFrame([candidate.__dict__ for candidate in selected]).to_csv(discovery_dir / "selected.csv", index=False)


def main() -> int:
    try:
        args = parse_args()
        if args.max_symbols <= 0:
            raise ValueError("--max-symbols must be positive.")
        slug = args.slug or slugify(args.theme_name)
        output = args.output or THEME_DIR / f"{slug}.yaml"
        seed_symbols = []
        seen_seeds = set()
        for symbol in ([] if args.no_built_in_hints else built_in_seed_symbols(args.theme_name)) + (args.seed_symbols or []):
            normalized = normalize_symbol(symbol)
            if normalized and normalized not in seen_seeds:
                seed_symbols.append(normalized)
                seen_seeds.add(normalized)
        allowed_quote_types = {quote_type.upper() for quote_type in args.quote_types}
        queries = build_queries(args.theme_name, args.queries)
        candidates = search_candidates(
            queries=queries,
            max_results_per_query=args.max_results_per_query,
            allowed_quote_types=allowed_quote_types,
            seed_symbols=seed_symbols,
        )
        candidates = validate_candidates(candidates, skip_validation=args.skip_validation)
        if not candidates:
            raise ValueError("No ticker candidates found. Try --queries or --seed-symbols.")
        selected = candidates[: args.max_symbols]
        benchmark = choose_benchmark(selected, args.benchmark)
        leader = choose_leader(selected, benchmark, args.leader)
        write_outputs(args, candidates, selected, benchmark, leader, slug, output, queries)

        print(f"Wrote {output}")
        print(f"Wrote discovery tables under {DISCOVERY_DIR / slug}")
        print("\nSelected symbols:")
        print(", ".join(candidate.symbol for candidate in selected))
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
