from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


DEFAULT_PRICES = Path("data/prices/etf_daily.parquet")
DEFAULT_REPORT_DIR = Path("reports")
TRADING_DAYS = 252


@dataclass(frozen=True)
class AnalysisConfig:
    prices_path: Path
    report_dir: Path
    risk_free_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze daily ETF prices and generate basic quant research reports."
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=DEFAULT_PRICES,
        help="Input parquet file produced by download_prices.py.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for generated reports.",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Annual risk-free rate used in Sharpe calculation, e.g. 0.04.",
    )
    return parser.parse_args()


def load_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")

    prices = pd.read_parquet(path)
    required_columns = {"date", "symbol", "adj_close"}
    missing_columns = required_columns - set(prices.columns)
    if missing_columns:
        raise ValueError(f"Price file is missing columns: {sorted(missing_columns)}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["symbol"] = prices["symbol"].astype(str)
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    return prices.sort_values(["symbol", "date"]).reset_index(drop=True)


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, object]] = []
    for symbol, frame in prices.groupby("symbol", sort=True):
        frame = frame.sort_values("date")
        duplicate_dates = int(frame["date"].duplicated().sum())
        missing_adj_close = int(frame["adj_close"].isna().sum())
        non_positive_adj_close = int((frame["adj_close"] <= 0).sum())
        checks.append(
            {
                "symbol": symbol,
                "rows": len(frame),
                "first_date": frame["date"].min().date(),
                "last_date": frame["date"].max().date(),
                "duplicate_dates": duplicate_dates,
                "missing_adj_close": missing_adj_close,
                "non_positive_adj_close": non_positive_adj_close,
            }
        )

    quality = pd.DataFrame(checks)
    bad_rows = quality[
        (quality["duplicate_dates"] > 0)
        | (quality["missing_adj_close"] > 0)
        | (quality["non_positive_adj_close"] > 0)
    ]
    if not bad_rows.empty:
        raise ValueError(f"Price quality checks failed:\n{bad_rows.to_string(index=False)}")
    return quality


def build_price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    matrix = prices.pivot(index="date", columns="symbol", values="adj_close")
    matrix = matrix.sort_index()

    missing = matrix.isna().sum()
    if missing.any():
        logging.warning("Filling missing adjusted closes with forward-fill then backward-fill.")
        logging.warning("\n%s", missing[missing > 0].to_string())
        matrix = matrix.ffill().bfill()

    return matrix


def max_drawdown(wealth: pd.Series) -> float:
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min())


def summarize_returns(returns: pd.DataFrame, risk_free_rate: float) -> pd.DataFrame:
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0
    excess_returns = returns - daily_rf
    wealth = (1.0 + returns).cumprod()

    rows: list[dict[str, float | str | int]] = []
    for symbol in returns.columns:
        series = returns[symbol].dropna()
        symbol_wealth = wealth[symbol].dropna()
        if series.empty:
            continue

        years = len(series) / TRADING_DAYS
        total_return = float(symbol_wealth.iloc[-1] - 1.0)
        annual_return = float(symbol_wealth.iloc[-1] ** (1.0 / years) - 1.0)
        annual_volatility = float(series.std(ddof=1) * (TRADING_DAYS**0.5))
        annual_excess_return = float(excess_returns[symbol].dropna().mean() * TRADING_DAYS)
        sharpe = annual_excess_return / annual_volatility if annual_volatility > 0 else float("nan")
        mdd = max_drawdown(symbol_wealth)
        calmar = annual_return / abs(mdd) if mdd < 0 else float("nan")

        rows.append(
            {
                "symbol": symbol,
                "observations": len(series),
                "total_return": total_return,
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe": sharpe,
                "max_drawdown": mdd,
                "calmar": calmar,
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values("sharpe", ascending=False).reset_index(drop=True)


def add_equal_weight_portfolio(returns: pd.DataFrame) -> pd.DataFrame:
    output = returns.copy()
    output["EW_PORTFOLIO"] = returns.mean(axis=1)
    return output


def write_csv_reports(
    quality: pd.DataFrame,
    summary: pd.DataFrame,
    returns: pd.DataFrame,
    wealth: pd.DataFrame,
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    quality.to_csv(report_dir / "data_quality.csv", index=False)
    summary.to_csv(report_dir / "performance_summary.csv", index=False)
    returns.to_csv(report_dir / "daily_returns.csv", index_label="date")
    wealth.to_csv(report_dir / "cumulative_wealth.csv", index_label="date")


def write_equity_curve_html(wealth: pd.DataFrame, report_dir: Path) -> None:
    fig = go.Figure()
    for symbol in wealth.columns:
        line_width = 3 if symbol == "EW_PORTFOLIO" else 1
        fig.add_trace(
            go.Scatter(
                x=wealth.index,
                y=wealth[symbol],
                mode="lines",
                name=symbol,
                line={"width": line_width},
            )
        )

    fig.update_layout(
        title="ETF Cumulative Wealth",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=720,
    )
    fig.write_html(report_dir / "equity_curves.html", include_plotlyjs="cdn")


def print_console_summary(summary: pd.DataFrame) -> None:
    display_columns = [
        "symbol",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
    ]
    formatted = summary[display_columns].copy()
    percent_columns = ["annual_return", "annual_volatility", "max_drawdown"]
    for column in percent_columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")
    for column in ["sharpe", "calmar"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2f}")

    print("\nTop assets by Sharpe:")
    print(formatted.head(10).to_string(index=False))


def run_analysis(config: AnalysisConfig) -> None:
    prices = load_prices(config.prices_path)
    quality = validate_prices(prices)
    price_matrix = build_price_matrix(prices)
    returns = price_matrix.pct_change().dropna(how="all")
    returns = add_equal_weight_portfolio(returns)
    wealth = (1.0 + returns).cumprod()
    summary = summarize_returns(returns, config.risk_free_rate)

    write_csv_reports(quality, summary, returns, wealth, config.report_dir)
    write_equity_curve_html(wealth, config.report_dir)
    print_console_summary(summary)

    logging.info("Read %d rows from %s", len(prices), config.prices_path)
    logging.info("Analyzed %d trading days and %d return series", len(returns), len(returns.columns))
    logging.info("Wrote reports to %s", config.report_dir)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = AnalysisConfig(
        prices_path=args.prices,
        report_dir=args.report_dir,
        risk_free_rate=args.risk_free_rate,
    )

    try:
        run_analysis(config)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
