from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from analyze_prices import TRADING_DAYS, build_price_matrix, load_prices, max_drawdown


DEFAULT_PRICES = Path("data/prices/ai_chip_daily.parquet")
DEFAULT_REPORT_DIR = Path("reports/ai_chip_trends")
THEME_BENCHMARK = "SMH"
LEADER_SYMBOL = "NVDA"

SYMBOL_LABELS = {
    "AMD": "GPU / CPU",
    "AMAT": "Semiconductor equipment",
    "ARM": "IP / CPU architecture",
    "ASML": "Lithography equipment",
    "AVGO": "AI networking / ASIC",
    "INTC": "CPU / foundry",
    "KLAC": "Process control equipment",
    "LRCX": "Semiconductor equipment",
    "MRVL": "AI networking / custom silicon",
    "MU": "Memory",
    "NVDA": "AI accelerator leader",
    "QCOM": "Mobile / edge AI chips",
    "SMH": "Semiconductor ETF",
    "SOXX": "Semiconductor ETF",
    "TSM": "Foundry",
}


@dataclass(frozen=True)
class AiChipTrendConfig:
    prices_path: Path
    report_dir: Path
    lookback_days: int
    short_days: int
    medium_days: int
    half_year_days: int
    year_days: int
    volatility_days: int
    correlation_days: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze recent AI semiconductor stock and ETF trends.")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES, help="Input parquet file.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for generated reports.")
    parser.add_argument("--lookback-days", type=int, default=504, help="Main trend window, roughly two trading years.")
    parser.add_argument("--short-days", type=int, default=21, help="Short return window, roughly one month.")
    parser.add_argument("--medium-days", type=int, default=63, help="Medium return window, roughly one quarter.")
    parser.add_argument("--half-year-days", type=int, default=126, help="Half-year return window.")
    parser.add_argument("--year-days", type=int, default=252, help="One-year return window.")
    parser.add_argument("--volatility-days", type=int, default=63, help="Window for recent realized volatility.")
    parser.add_argument("--correlation-days", type=int, default=126, help="Window for latest correlation matrix.")
    return parser.parse_args()


def validate_config(config: AiChipTrendConfig) -> None:
    fields = {
        "lookback_days": config.lookback_days,
        "short_days": config.short_days,
        "medium_days": config.medium_days,
        "half_year_days": config.half_year_days,
        "year_days": config.year_days,
        "volatility_days": config.volatility_days,
        "correlation_days": config.correlation_days,
    }
    invalid = [name for name, value in fields.items() if value <= 0]
    if invalid:
        raise ValueError(f"Expected positive windows, got invalid fields: {invalid}")


def return_over_window(price_matrix: pd.DataFrame, days: int) -> pd.DataFrame:
    return price_matrix / price_matrix.shift(days) - 1.0


def latest_value(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.iloc[-1])


def annualized_return(total_return: float, observations: int) -> float:
    years = observations / TRADING_DAYS
    if years <= 0 or total_return <= -1:
        return float("nan")
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def beta_to_benchmark(returns: pd.DataFrame, symbol: str, benchmark: str) -> float:
    if symbol == benchmark or benchmark not in returns.columns or symbol not in returns.columns:
        return float("nan")
    sample = returns[[symbol, benchmark]].dropna()
    if len(sample) < 30:
        return float("nan")
    variance = sample[benchmark].var(ddof=1)
    if variance <= 0:
        return float("nan")
    return float(sample[symbol].cov(sample[benchmark]) / variance)


def build_snapshot(price_matrix: pd.DataFrame, config: AiChipTrendConfig) -> pd.DataFrame:
    window = price_matrix.tail(config.lookback_days + 1)
    returns = price_matrix.pct_change()
    recent_returns = returns.tail(config.correlation_days)
    ret_1m = return_over_window(price_matrix, config.short_days)
    ret_3m = return_over_window(price_matrix, config.medium_days)
    ret_6m = return_over_window(price_matrix, config.half_year_days)
    ret_1y = return_over_window(price_matrix, config.year_days)
    vol = returns.rolling(config.volatility_days).std() * (TRADING_DAYS**0.5)
    highs_1y = price_matrix.rolling(config.year_days).max()

    rows: list[dict[str, object]] = []
    for symbol in price_matrix.columns:
        series = window[symbol].dropna()
        if len(series) < 2:
            continue
        wealth = series / series.iloc[0]
        total_return = float(wealth.iloc[-1] - 1.0)
        symbol_returns = returns[symbol].reindex(series.index).dropna()
        realized_vol = float(symbol_returns.std(ddof=1) * (TRADING_DAYS**0.5)) if len(symbol_returns) > 2 else float("nan")
        ann_return = annualized_return(total_return, len(symbol_returns))
        drawdown = max_drawdown(wealth)
        row = {
            "symbol": symbol,
            "label": SYMBOL_LABELS.get(symbol, symbol),
            "start_date": series.index[0].date(),
            "end_date": series.index[-1].date(),
            "observations": len(symbol_returns),
            "total_return_2y": total_return,
            "annual_return_2y": ann_return,
            "annual_volatility_2y": realized_vol,
            "sharpe_like_2y": ann_return / realized_vol if realized_vol and realized_vol > 0 else float("nan"),
            "max_drawdown_2y": drawdown,
            "return_1m": latest_value(ret_1m[symbol]),
            "return_3m": latest_value(ret_3m[symbol]),
            "return_6m": latest_value(ret_6m[symbol]),
            "return_1y": latest_value(ret_1y[symbol]),
            "volatility_3m": latest_value(vol[symbol]),
            "distance_from_52w_high": latest_value(price_matrix[symbol] / highs_1y[symbol] - 1.0),
            "correlation_to_smh": float(recent_returns[symbol].corr(recent_returns[THEME_BENCHMARK]))
            if THEME_BENCHMARK in recent_returns.columns and symbol != THEME_BENCHMARK
            else float("nan"),
            "correlation_to_nvda": float(recent_returns[symbol].corr(recent_returns[LEADER_SYMBOL]))
            if LEADER_SYMBOL in recent_returns.columns and symbol != LEADER_SYMBOL
            else float("nan"),
            "beta_to_smh": beta_to_benchmark(recent_returns, symbol, THEME_BENCHMARK),
        }
        rows.append(row)

    snapshot = pd.DataFrame(rows)
    if snapshot.empty:
        return snapshot

    snapshot["trend_score"] = (
        snapshot["total_return_2y"].rank(pct=True)
        + snapshot["return_6m"].rank(pct=True)
        + snapshot["return_3m"].rank(pct=True)
        + snapshot["return_1m"].rank(pct=True) * 0.5
        - snapshot["volatility_3m"].rank(pct=True) * 0.4
        + snapshot["distance_from_52w_high"].rank(pct=True) * 0.3
    )
    return snapshot.sort_values("trend_score", ascending=False).reset_index(drop=True)


def build_group_summary(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot
    return (
        snapshot.groupby("label", as_index=False)
        .agg(
            symbols=("symbol", lambda values: ",".join(values)),
            count=("symbol", "count"),
            median_total_return_2y=("total_return_2y", "median"),
            median_return_6m=("return_6m", "median"),
            median_return_3m=("return_3m", "median"),
            median_volatility_3m=("volatility_3m", "median"),
            median_drawdown_2y=("max_drawdown_2y", "median"),
        )
        .sort_values("median_total_return_2y", ascending=False)
        .reset_index(drop=True)
    )


def build_correlation(price_matrix: pd.DataFrame, days: int) -> pd.DataFrame:
    returns = price_matrix.pct_change().tail(days)
    return returns.corr()


def write_normalized_performance(price_matrix: pd.DataFrame, config: AiChipTrendConfig, report_dir: Path) -> None:
    window = price_matrix.tail(config.lookback_days + 1)
    normalized = window / window.iloc[0]
    fig = go.Figure()
    snapshot = build_snapshot(price_matrix, config)
    leaders = set(snapshot.head(6)["symbol"]) if not snapshot.empty else set(normalized.columns)
    for symbol in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[symbol],
                mode="lines",
                name=symbol,
                line={"width": 3 if symbol in leaders else 1},
                opacity=1.0 if symbol in leaders else 0.45,
            )
        )
    fig.update_layout(
        title="AI Chip Theme: Growth of $1",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=680,
    )
    fig.write_html(report_dir / "normalized_performance.html", include_plotlyjs="cdn")


def write_return_bars(snapshot: pd.DataFrame, report_dir: Path) -> None:
    ordered = snapshot.sort_values("total_return_2y", ascending=False)
    fig = go.Figure()
    for column, label in [
        ("return_3m", "3M"),
        ("return_6m", "6M"),
        ("return_1y", "1Y"),
        ("total_return_2y", "2Y"),
    ]:
        fig.add_trace(go.Bar(x=ordered["symbol"], y=ordered[column], name=label))
    fig.update_layout(
        title="AI Chip Theme Returns By Horizon",
        xaxis_title="Symbol",
        yaxis_title="Return",
        barmode="group",
        template="plotly_white",
        width=1200,
        height=620,
    )
    fig.write_html(report_dir / "return_horizon_bars.html", include_plotlyjs="cdn")


def write_drawdown_chart(price_matrix: pd.DataFrame, config: AiChipTrendConfig, report_dir: Path) -> None:
    window = price_matrix.tail(config.lookback_days + 1)
    wealth = window / window.iloc[0]
    drawdown = wealth / wealth.cummax() - 1.0
    fig = go.Figure()
    for symbol in drawdown.columns:
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown[symbol], mode="lines", name=symbol))
    fig.update_layout(
        title="AI Chip Theme Drawdowns",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=620,
    )
    fig.write_html(report_dir / "drawdowns.html", include_plotlyjs="cdn")


def write_correlation_heatmap(correlation: pd.DataFrame, report_dir: Path) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            x=correlation.columns,
            y=correlation.index,
            z=correlation.values,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar={"title": "Corr"},
        )
    )
    fig.update_layout(
        title="Latest AI Chip Theme Correlation",
        xaxis_title="Symbol",
        yaxis_title="Symbol",
        template="plotly_white",
        width=900,
        height=820,
    )
    fig.write_html(report_dir / "correlation_heatmap.html", include_plotlyjs="cdn")


def write_strength_heatmap(snapshot: pd.DataFrame, report_dir: Path) -> None:
    columns = ["return_1m", "return_3m", "return_6m", "return_1y", "total_return_2y", "volatility_3m", "max_drawdown_2y"]
    heat = snapshot.set_index("symbol")[columns]
    fig = go.Figure(
        data=go.Heatmap(
            x=heat.columns,
            y=heat.index,
            z=heat.values,
            colorscale="RdYlGn",
            colorbar={"title": "Value"},
        )
    )
    fig.update_layout(
        title="Relative Strength And Risk Snapshot",
        xaxis_title="Metric",
        yaxis_title="Symbol",
        template="plotly_white",
        width=1000,
        height=760,
    )
    fig.write_html(report_dir / "strength_heatmap.html", include_plotlyjs="cdn")


def format_pct(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def write_markdown_summary(snapshot: pd.DataFrame, group_summary: pd.DataFrame, report_dir: Path) -> None:
    leaders = snapshot.head(5)
    laggards = snapshot.tail(5).sort_values("trend_score")
    benchmark = snapshot.loc[snapshot["symbol"] == THEME_BENCHMARK]
    benchmark_text = ""
    if not benchmark.empty:
        row = benchmark.iloc[0]
        benchmark_text = (
            f"- {THEME_BENCHMARK} 2Y return: {format_pct(row['total_return_2y'])}; "
            f"3M return: {format_pct(row['return_3m'])}; max drawdown: {format_pct(row['max_drawdown_2y'])}"
        )

    lines = [
        "# AI Chip Trend Report",
        "",
        f"- End date: {snapshot['end_date'].max()}",
        f"- Symbols: {', '.join(snapshot['symbol'])}",
        benchmark_text,
        "",
        "## Trend Leaders",
        "",
    ]
    lines.extend(
        f"- {row.symbol}: 2Y={format_pct(row.total_return_2y)}, 6M={format_pct(row.return_6m)}, "
        f"3M={format_pct(row.return_3m)}, drawdown={format_pct(row.max_drawdown_2y)}, {row.label}"
        for row in leaders.itertuples(index=False)
    )
    lines.extend(["", "## Trend Laggards", ""])
    lines.extend(
        f"- {row.symbol}: 2Y={format_pct(row.total_return_2y)}, 6M={format_pct(row.return_6m)}, "
        f"3M={format_pct(row.return_3m)}, drawdown={format_pct(row.max_drawdown_2y)}, {row.label}"
        for row in laggards.itertuples(index=False)
    )
    if not group_summary.empty:
        lines.extend(["", "## Industry Chain Snapshot", ""])
        lines.extend(
            f"- {row.label}: symbols={row.symbols}, median 2Y={format_pct(row.median_total_return_2y)}, "
            f"median 3M={format_pct(row.median_return_3m)}"
            for row in group_summary.head(8).itertuples(index=False)
        )
    lines.extend(
        [
            "",
            "## How To Read",
            "",
            "- SMH and SOXX are theme benchmarks; use them to separate stock-specific moves from broad semiconductor beta.",
            "- Strong 2-year returns with shallow distance from the 52-week high indicate persistent leadership.",
            "- High return with high volatility or deep drawdown is a more fragile trend.",
            "- Correlation to NVDA helps show whether a name is trading as part of the AI leader complex.",
            "",
        ]
    )
    (report_dir / "ai_chip_summary.md").write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")


def write_reports(
    config: AiChipTrendConfig,
    price_matrix: pd.DataFrame,
    snapshot: pd.DataFrame,
    group_summary: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    config.report_dir.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(config.report_dir / "trend_snapshot.csv", index=False)
    group_summary.to_csv(config.report_dir / "group_summary.csv", index=False)
    correlation.to_csv(config.report_dir / "latest_correlation.csv")
    pd.DataFrame(
        [
            {"key": "lookback_days", "value": config.lookback_days},
            {"key": "short_days", "value": config.short_days},
            {"key": "medium_days", "value": config.medium_days},
            {"key": "half_year_days", "value": config.half_year_days},
            {"key": "year_days", "value": config.year_days},
            {"key": "volatility_days", "value": config.volatility_days},
            {"key": "correlation_days", "value": config.correlation_days},
        ]
    ).to_csv(config.report_dir / "ai_chip_trend_config.csv", index=False)

    write_normalized_performance(price_matrix, config, config.report_dir)
    write_return_bars(snapshot, config.report_dir)
    write_drawdown_chart(price_matrix, config, config.report_dir)
    write_correlation_heatmap(correlation, config.report_dir)
    write_strength_heatmap(snapshot, config.report_dir)
    write_markdown_summary(snapshot, group_summary, config.report_dir)


def print_summary(snapshot: pd.DataFrame) -> None:
    print("\nAI chip trend leaders:")
    display = snapshot[
        [
            "symbol",
            "label",
            "total_return_2y",
            "return_6m",
            "return_3m",
            "volatility_3m",
            "max_drawdown_2y",
            "distance_from_52w_high",
        ]
    ].head(10).copy()
    for column in ["total_return_2y", "return_6m", "return_3m", "volatility_3m", "max_drawdown_2y", "distance_from_52w_high"]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    print(display.to_string(index=False))


def run_ai_chip_trends(config: AiChipTrendConfig) -> None:
    validate_config(config)
    prices = load_prices(config.prices_path)
    price_matrix = build_price_matrix(prices)
    if price_matrix.empty:
        raise ValueError("No price data available.")
    snapshot = build_snapshot(price_matrix, config)
    if snapshot.empty:
        raise ValueError("Not enough data to build trend snapshot.")
    group_summary = build_group_summary(snapshot)
    correlation = build_correlation(price_matrix, config.correlation_days)
    write_reports(config, price_matrix, snapshot, group_summary, correlation)
    print_summary(snapshot)
    logging.info("Read %d rows from %s", len(prices), config.prices_path)
    logging.info("Wrote AI chip trend report to %s", config.report_dir)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = AiChipTrendConfig(
        prices_path=args.prices,
        report_dir=args.report_dir,
        lookback_days=args.lookback_days,
        short_days=args.short_days,
        medium_days=args.medium_days,
        half_year_days=args.half_year_days,
        year_days=args.year_days,
        volatility_days=args.volatility_days,
        correlation_days=args.correlation_days,
    )
    try:
        run_ai_chip_trends(config)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
