from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from analyze_prices import TRADING_DAYS, build_price_matrix, load_prices


DEFAULT_PRICES = Path("data/prices/etf_daily.parquet")
DEFAULT_REPORT_DIR = Path("reports/market_regime")

ASSET_LABELS = {
    "SPY": "US large-cap equity",
    "QQQ": "US growth/technology",
    "IWM": "US small-cap equity",
    "DIA": "US blue-chip equity",
    "EFA": "Developed ex-US equity",
    "EEM": "Emerging-market equity",
    "TLT": "Long-duration Treasury",
    "IEF": "Intermediate Treasury",
    "SHY": "Short Treasury / cash proxy",
    "GLD": "Gold",
    "SLV": "Silver",
    "USO": "Crude oil",
    "VNQ": "Real estate",
    "XLE": "Energy sector",
    "XLF": "Financial sector",
    "XLI": "Industrial sector",
    "XLK": "Technology sector",
    "XLP": "Consumer staples sector",
    "XLU": "Utilities sector",
    "XLV": "Health care sector",
    "XLY": "Consumer discretionary sector",
}

SECTOR_SYMBOLS = ["XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
CORE_SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "SHY", "GLD", "SLV", "USO", "VNQ"]


@dataclass(frozen=True)
class MarketRegimeConfig:
    prices_path: Path
    report_dir: Path
    trend_short_days: int
    trend_medium_days: int
    trend_long_days: int
    volatility_days: int
    drawdown_days: int
    correlation_days: int
    recent_days: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze current ETF market regime and cross-asset conditions.")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES, help="Input parquet file from download_prices.py.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for generated reports.")
    parser.add_argument("--trend-short-days", type=int, default=21, help="Short trend window, roughly one month.")
    parser.add_argument("--trend-medium-days", type=int, default=63, help="Medium trend window, roughly one quarter.")
    parser.add_argument("--trend-long-days", type=int, default=126, help="Long trend window, roughly six months.")
    parser.add_argument("--volatility-days", type=int, default=21, help="Window for recent realized volatility.")
    parser.add_argument("--drawdown-days", type=int, default=63, help="Window for recent drawdown.")
    parser.add_argument("--correlation-days", type=int, default=63, help="Window for latest correlation matrix.")
    parser.add_argument("--recent-days", type=int, default=252, help="History length used in time-series charts.")
    return parser.parse_args()


def validate_config(config: MarketRegimeConfig) -> None:
    positive_fields = {
        "trend_short_days": config.trend_short_days,
        "trend_medium_days": config.trend_medium_days,
        "trend_long_days": config.trend_long_days,
        "volatility_days": config.volatility_days,
        "drawdown_days": config.drawdown_days,
        "correlation_days": config.correlation_days,
        "recent_days": config.recent_days,
    }
    bad = [name for name, value in positive_fields.items() if value <= 0]
    if bad:
        raise ValueError(f"Expected positive window sizes, got invalid fields: {bad}")


def require_symbols(price_matrix: pd.DataFrame, symbols: list[str]) -> list[str]:
    return [symbol for symbol in symbols if symbol in price_matrix.columns]


def return_over_window(price_matrix: pd.DataFrame, days: int) -> pd.DataFrame:
    return price_matrix / price_matrix.shift(days) - 1.0


def rolling_max_drawdown(price_matrix: pd.DataFrame, days: int) -> pd.DataFrame:
    rolling_peak = price_matrix.rolling(days).max()
    return price_matrix / rolling_peak - 1.0


def zscore(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    mean = frame.rolling(days).mean()
    std = frame.rolling(days).std(ddof=1)
    return (frame - mean) / std


def build_indicator_history(price_matrix: pd.DataFrame, config: MarketRegimeConfig) -> pd.DataFrame:
    returns = price_matrix.pct_change()
    indicators = pd.DataFrame(index=price_matrix.index)

    if "SPY" in price_matrix.columns:
        spy = price_matrix["SPY"]
        indicators["spy_above_200d"] = (spy > spy.rolling(200).mean()).astype(float)
        indicators["spy_1m_return"] = spy / spy.shift(config.trend_short_days) - 1.0
        indicators["spy_3m_return"] = spy / spy.shift(config.trend_medium_days) - 1.0
        indicators["spy_6m_return"] = spy / spy.shift(config.trend_long_days) - 1.0
        indicators["spy_1m_volatility"] = returns["SPY"].rolling(config.volatility_days).std() * (TRADING_DAYS**0.5)
        indicators["spy_3m_drawdown"] = rolling_max_drawdown(price_matrix[["SPY"]], config.drawdown_days)["SPY"]

    if {"SPY", "TLT"}.issubset(price_matrix.columns):
        indicators["spy_minus_tlt_3m"] = return_over_window(price_matrix, config.trend_medium_days)["SPY"] - return_over_window(
            price_matrix, config.trend_medium_days
        )["TLT"]

    if {"QQQ", "SPY"}.issubset(price_matrix.columns):
        indicators["qqq_minus_spy_3m"] = return_over_window(price_matrix, config.trend_medium_days)["QQQ"] - return_over_window(
            price_matrix, config.trend_medium_days
        )["SPY"]

    if {"IWM", "SPY"}.issubset(price_matrix.columns):
        indicators["iwm_minus_spy_3m"] = return_over_window(price_matrix, config.trend_medium_days)["IWM"] - return_over_window(
            price_matrix, config.trend_medium_days
        )["SPY"]

    if {"GLD", "SPY"}.issubset(price_matrix.columns):
        indicators["gld_minus_spy_3m"] = return_over_window(price_matrix, config.trend_medium_days)["GLD"] - return_over_window(
            price_matrix, config.trend_medium_days
        )["SPY"]

    if {"XLY", "XLP"}.issubset(price_matrix.columns):
        indicators["cyclical_minus_defensive_3m"] = (
            return_over_window(price_matrix, config.trend_medium_days)["XLY"]
            - return_over_window(price_matrix, config.trend_medium_days)["XLP"]
        )

    if {"XLE", "XLU"}.issubset(price_matrix.columns):
        indicators["energy_minus_utilities_3m"] = (
            return_over_window(price_matrix, config.trend_medium_days)["XLE"]
            - return_over_window(price_matrix, config.trend_medium_days)["XLU"]
        )

    return indicators.dropna(how="all")


def latest_value(series: pd.Series, default: float = float("nan")) -> float:
    clean = series.dropna()
    if clean.empty:
        return default
    return float(clean.iloc[-1])


def score_market_state(indicators: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = indicators.iloc[-1]
    risk_score = 0
    notes: list[str] = []

    if latest.get("spy_above_200d", 0.0) >= 1.0:
        risk_score += 2
        notes.append("SPY is above its 200-day moving average.")
    else:
        risk_score -= 2
        notes.append("SPY is below its 200-day moving average.")

    if latest.get("spy_3m_return", 0.0) > 0:
        risk_score += 1
        notes.append("US equity has positive 3-month momentum.")
    else:
        risk_score -= 1
        notes.append("US equity has weak or negative 3-month momentum.")

    if latest.get("spy_minus_tlt_3m", 0.0) > 0:
        risk_score += 1
        notes.append("Equity is outperforming long Treasuries.")
    else:
        risk_score -= 1
        notes.append("Long Treasuries are outperforming equity.")

    if latest.get("cyclical_minus_defensive_3m", 0.0) > 0:
        risk_score += 1
        notes.append("Cyclical sectors are ahead of defensive sectors.")
    else:
        risk_score -= 1
        notes.append("Defensive sectors are ahead of cyclicals.")

    if latest.get("spy_1m_volatility", 0.0) > 0.25:
        risk_score -= 1
        notes.append("Realized equity volatility is elevated.")
    elif latest.get("spy_1m_volatility", 0.0) < 0.15:
        risk_score += 1
        notes.append("Realized equity volatility is contained.")

    drawdown = latest.get("spy_3m_drawdown", 0.0)
    if drawdown < -0.10:
        risk_score -= 2
        notes.append("SPY is in a meaningful recent drawdown.")
    elif drawdown > -0.03:
        risk_score += 1
        notes.append("Recent SPY drawdown is shallow.")

    if risk_score >= 3:
        regime = "risk_on"
        posture = "Offensive"
    elif risk_score <= -3:
        regime = "risk_off"
        posture = "Defensive"
    else:
        regime = "mixed"
        posture = "Balanced"

    if latest.get("spy_3m_return", 0.0) > 0 and latest.get("spy_1m_volatility", 0.0) < 0.20:
        trend_state = "uptrend"
    elif latest.get("spy_3m_return", 0.0) < 0 and latest.get("spy_1m_volatility", 0.0) > 0.20:
        trend_state = "stress"
    else:
        trend_state = "choppy"

    state = pd.DataFrame(
        [
            {
                "date": indicators.index[-1].date(),
                "regime": regime,
                "posture": posture,
                "trend_state": trend_state,
                "risk_score": risk_score,
                "spy_above_200d": bool(latest.get("spy_above_200d", 0.0) >= 1.0),
                "spy_1m_return": latest.get("spy_1m_return"),
                "spy_3m_return": latest.get("spy_3m_return"),
                "spy_6m_return": latest.get("spy_6m_return"),
                "spy_1m_volatility": latest.get("spy_1m_volatility"),
                "spy_3m_drawdown": latest.get("spy_3m_drawdown"),
                "spy_minus_tlt_3m": latest.get("spy_minus_tlt_3m"),
                "qqq_minus_spy_3m": latest.get("qqq_minus_spy_3m"),
                "iwm_minus_spy_3m": latest.get("iwm_minus_spy_3m"),
                "gld_minus_spy_3m": latest.get("gld_minus_spy_3m"),
                "cyclical_minus_defensive_3m": latest.get("cyclical_minus_defensive_3m"),
                "energy_minus_utilities_3m": latest.get("energy_minus_utilities_3m"),
            }
        ]
    )
    note_frame = pd.DataFrame({"note": notes})
    return state, note_frame


def build_regime_history(indicators: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_value, row in indicators.dropna(subset=["spy_3m_return", "spy_1m_volatility"], how="all").iterrows():
        score = 0
        score += 2 if row.get("spy_above_200d", 0.0) >= 1.0 else -2
        score += 1 if row.get("spy_3m_return", 0.0) > 0 else -1
        score += 1 if row.get("spy_minus_tlt_3m", 0.0) > 0 else -1
        score += 1 if row.get("cyclical_minus_defensive_3m", 0.0) > 0 else -1
        score += -1 if row.get("spy_1m_volatility", 0.0) > 0.25 else 0
        score += -2 if row.get("spy_3m_drawdown", 0.0) < -0.10 else 0
        if score >= 3:
            regime = "risk_on"
        elif score <= -3:
            regime = "risk_off"
        else:
            regime = "mixed"
        rows.append({"date": date_value.date(), "risk_score": score, "regime": regime})
    return pd.DataFrame(rows)


def build_asset_snapshot(price_matrix: pd.DataFrame, config: MarketRegimeConfig) -> pd.DataFrame:
    returns = price_matrix.pct_change()
    ret_1m = return_over_window(price_matrix, config.trend_short_days)
    ret_3m = return_over_window(price_matrix, config.trend_medium_days)
    ret_6m = return_over_window(price_matrix, config.trend_long_days)
    vol_1m = returns.rolling(config.volatility_days).std() * (TRADING_DAYS**0.5)
    dd_3m = rolling_max_drawdown(price_matrix, config.drawdown_days)
    above_200d = price_matrix > price_matrix.rolling(200).mean()

    rows: list[dict[str, object]] = []
    latest_date = price_matrix.index[-1].date()
    for symbol in price_matrix.columns:
        rows.append(
            {
                "date": latest_date,
                "symbol": symbol,
                "label": ASSET_LABELS.get(symbol, symbol),
                "return_1m": latest_value(ret_1m[symbol]),
                "return_3m": latest_value(ret_3m[symbol]),
                "return_6m": latest_value(ret_6m[symbol]),
                "volatility_1m": latest_value(vol_1m[symbol]),
                "drawdown_3m": latest_value(dd_3m[symbol]),
                "above_200d": bool(above_200d[symbol].dropna().iloc[-1]) if not above_200d[symbol].dropna().empty else False,
            }
        )
    snapshot = pd.DataFrame(rows)
    snapshot["trend_score"] = (
        snapshot["return_1m"].rank(pct=True)
        + snapshot["return_3m"].rank(pct=True)
        + snapshot["return_6m"].rank(pct=True)
        - snapshot["volatility_1m"].rank(pct=True) * 0.5
    )
    return snapshot.sort_values("trend_score", ascending=False).reset_index(drop=True)


def build_relative_strength(snapshot: pd.DataFrame) -> pd.DataFrame:
    columns = ["symbol", "label", "return_1m", "return_3m", "return_6m", "volatility_1m", "drawdown_3m", "trend_score"]
    return snapshot[columns].sort_values("trend_score", ascending=False).reset_index(drop=True)


def build_latest_correlation(price_matrix: pd.DataFrame, days: int) -> pd.DataFrame:
    returns = price_matrix.pct_change()
    symbols = require_symbols(price_matrix, CORE_SYMBOLS)
    if len(symbols) < 2:
        symbols = list(price_matrix.columns)
    return returns[symbols].tail(days).corr()


def write_regime_timeline(regime_history: pd.DataFrame, report_dir: Path) -> None:
    fig = go.Figure()
    colors = {"risk_on": "#16a34a", "mixed": "#64748b", "risk_off": "#dc2626"}
    for regime, frame in regime_history.groupby("regime"):
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["risk_score"],
                mode="markers",
                name=regime,
                marker={"color": colors.get(regime, "#64748b"), "size": 6},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=regime_history["date"],
            y=regime_history["risk_score"],
            mode="lines",
            name="risk_score",
            line={"color": "#0f172a", "width": 1},
        )
    )
    fig.update_layout(
        title="Market Regime Score",
        xaxis_title="Date",
        yaxis_title="Risk score",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=640,
    )
    fig.write_html(report_dir / "regime_timeline.html", include_plotlyjs="cdn")


def write_asset_trend_chart(snapshot: pd.DataFrame, report_dir: Path) -> None:
    top = snapshot.sort_values("return_3m", ascending=False)
    fig = go.Figure(
        data=[
            go.Bar(
                x=top["symbol"],
                y=top["return_3m"],
                marker_color=["#16a34a" if value >= 0 else "#dc2626" for value in top["return_3m"]],
                text=top["label"],
                hovertemplate="%{x}<br>%{text}<br>3M return=%{y:.2%}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Latest 3-Month Cross-Asset Returns",
        xaxis_title="Symbol",
        yaxis_title="3-month return",
        template="plotly_white",
        width=1200,
        height=600,
    )
    fig.write_html(report_dir / "asset_trends.html", include_plotlyjs="cdn")


def write_indicator_chart(indicators: pd.DataFrame, config: MarketRegimeConfig, report_dir: Path) -> None:
    recent = indicators.tail(config.recent_days)
    fig = go.Figure()
    for column in [
        "spy_3m_return",
        "spy_minus_tlt_3m",
        "qqq_minus_spy_3m",
        "gld_minus_spy_3m",
        "cyclical_minus_defensive_3m",
    ]:
        if column in recent.columns:
            fig.add_trace(go.Scatter(x=recent.index, y=recent[column], mode="lines", name=column))
    fig.update_layout(
        title="Market Condition Indicators",
        xaxis_title="Date",
        yaxis_title="Return spread / momentum",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=640,
    )
    fig.write_html(report_dir / "market_indicators.html", include_plotlyjs="cdn")


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
        title="Latest Cross-Asset Correlation",
        xaxis_title="Symbol",
        yaxis_title="Symbol",
        template="plotly_white",
        width=900,
        height=760,
    )
    fig.write_html(report_dir / "correlation_heatmap.html", include_plotlyjs="cdn")


def format_pct(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def write_markdown_summary(state: pd.DataFrame, notes: pd.DataFrame, relative_strength: pd.DataFrame, report_dir: Path) -> None:
    current = state.iloc[0]
    leaders = relative_strength.head(5)
    laggards = relative_strength.tail(5).sort_values("trend_score")
    lines = [
        "# Market Regime Report",
        "",
        f"- Date: {current['date']}",
        f"- Regime: {current['regime']}",
        f"- Suggested posture: {current['posture']}",
        f"- Trend state: {current['trend_state']}",
        f"- Risk score: {current['risk_score']}",
        f"- SPY 1M / 3M / 6M: {format_pct(current['spy_1m_return'])} / {format_pct(current['spy_3m_return'])} / {format_pct(current['spy_6m_return'])}",
        f"- SPY 1M vol: {format_pct(current['spy_1m_volatility'])}",
        f"- SPY recent drawdown: {format_pct(current['spy_3m_drawdown'])}",
        "",
        "## Diagnostics",
        "",
    ]
    lines.extend(f"- {note}" for note in notes["note"])
    lines.extend(["", "## Relative Strength Leaders", ""])
    lines.extend(
        f"- {row.symbol}: 3M={format_pct(row.return_3m)}, 1M vol={format_pct(row.volatility_1m)}, {row.label}"
        for row in leaders.itertuples(index=False)
    )
    lines.extend(["", "## Relative Strength Laggards", ""])
    lines.extend(
        f"- {row.symbol}: 3M={format_pct(row.return_3m)}, 1M vol={format_pct(row.volatility_1m)}, {row.label}"
        for row in laggards.itertuples(index=False)
    )
    lines.extend(
        [
            "",
            "## How To Use",
            "",
            "- Treat this as a decision-support screen, not an automatic buy/sell signal.",
            "- In risk-on regimes, inspect whether strategy holdings agree with the relative-strength leaders.",
            "- In risk-off regimes, check cash, Treasuries, gold, and defensive sectors before increasing equity exposure.",
            "- In mixed regimes, prefer smaller position sizes and require stronger factor evidence.",
            "",
        ]
    )
    (report_dir / "market_regime_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    config: MarketRegimeConfig,
    indicators: pd.DataFrame,
    state: pd.DataFrame,
    notes: pd.DataFrame,
    regime_history: pd.DataFrame,
    asset_snapshot: pd.DataFrame,
    relative_strength: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    config.report_dir.mkdir(parents=True, exist_ok=True)
    indicators.to_csv(config.report_dir / "market_indicators.csv", index_label="date")
    state.to_csv(config.report_dir / "market_state.csv", index=False)
    notes.to_csv(config.report_dir / "market_notes.csv", index=False)
    regime_history.to_csv(config.report_dir / "regime_history.csv", index=False)
    asset_snapshot.to_csv(config.report_dir / "asset_snapshot.csv", index=False)
    relative_strength.to_csv(config.report_dir / "relative_strength.csv", index=False)
    correlation.to_csv(config.report_dir / "latest_correlation.csv", index=True)
    pd.DataFrame(
        [
            {"key": "trend_short_days", "value": config.trend_short_days},
            {"key": "trend_medium_days", "value": config.trend_medium_days},
            {"key": "trend_long_days", "value": config.trend_long_days},
            {"key": "volatility_days", "value": config.volatility_days},
            {"key": "drawdown_days", "value": config.drawdown_days},
            {"key": "correlation_days", "value": config.correlation_days},
            {"key": "recent_days", "value": config.recent_days},
        ]
    ).to_csv(config.report_dir / "market_regime_config.csv", index=False)

    write_regime_timeline(regime_history, config.report_dir)
    write_asset_trend_chart(asset_snapshot, config.report_dir)
    write_indicator_chart(indicators, config, config.report_dir)
    write_correlation_heatmap(correlation, config.report_dir)
    write_markdown_summary(state, notes, relative_strength, config.report_dir)


def print_summary(state: pd.DataFrame, relative_strength: pd.DataFrame) -> None:
    current = state.iloc[0]
    print("\nMarket regime:")
    print(f"date: {current['date']}")
    print(f"regime: {current['regime']}")
    print(f"posture: {current['posture']}")
    print(f"risk_score: {current['risk_score']}")
    print("\nTop relative strength:")
    display = relative_strength[["symbol", "return_1m", "return_3m", "return_6m", "volatility_1m", "drawdown_3m"]].head(8).copy()
    for column in ["return_1m", "return_3m", "return_6m", "volatility_1m", "drawdown_3m"]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    print(display.to_string(index=False))


def run_market_regime(config: MarketRegimeConfig) -> None:
    validate_config(config)
    prices = load_prices(config.prices_path)
    price_matrix = build_price_matrix(prices)
    indicators = build_indicator_history(price_matrix, config)
    indicators = indicators.dropna(how="all")
    if indicators.empty:
        raise ValueError("Not enough data to compute market indicators.")

    state, notes = score_market_state(indicators)
    regime_history = build_regime_history(indicators)
    asset_snapshot = build_asset_snapshot(price_matrix, config)
    relative_strength = build_relative_strength(asset_snapshot)
    correlation = build_latest_correlation(price_matrix, config.correlation_days)

    write_reports(
        config=config,
        indicators=indicators,
        state=state,
        notes=notes,
        regime_history=regime_history,
        asset_snapshot=asset_snapshot,
        relative_strength=relative_strength,
        correlation=correlation,
    )
    print_summary(state, relative_strength)
    logging.info("Read %d rows from %s", len(prices), config.prices_path)
    logging.info("Wrote market regime report to %s", config.report_dir)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = MarketRegimeConfig(
        prices_path=args.prices,
        report_dir=args.report_dir,
        trend_short_days=args.trend_short_days,
        trend_medium_days=args.trend_medium_days,
        trend_long_days=args.trend_long_days,
        volatility_days=args.volatility_days,
        drawdown_days=args.drawdown_days,
        correlation_days=args.correlation_days,
        recent_days=args.recent_days,
    )

    try:
        run_market_regime(config)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
