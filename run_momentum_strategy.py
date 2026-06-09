from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from analyze_prices import TRADING_DAYS, build_price_matrix, load_prices, max_drawdown, summarize_returns


DEFAULT_PRICES = Path("data/prices/etf_daily.parquet")
DEFAULT_REPORT_DIR = Path("reports/momentum_rotation")


@dataclass(frozen=True)
class MomentumConfig:
    prices_path: Path
    report_dir: Path
    lookback_days: int
    top_k: int
    transaction_cost_bps: float
    risk_free_rate: float
    allow_negative_momentum: bool
    target_volatility: float | None
    vol_lookback_days: int
    max_leverage: float
    market_filter_symbol: str | None
    market_filter_ma_days: int
    symbols: list[str] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a monthly ETF momentum rotation strategy.")
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
        help="Directory for generated strategy reports.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=252,
        help="Momentum lookback window in trading days. 252 is roughly 12 months.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of ETFs to hold after each rebalance.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=5.0,
        help="One-way transaction cost in basis points.",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Annual risk-free rate used in Sharpe calculation, e.g. 0.04.",
    )
    parser.add_argument(
        "--allow-negative-momentum",
        action="store_true",
        help="Hold the top-ranked ETFs even when their momentum is negative.",
    )
    parser.add_argument(
        "--target-volatility",
        type=float,
        default=None,
        help="Optional annualized volatility target, e.g. 0.15. Exposure is scaled using trailing realized vol.",
    )
    parser.add_argument(
        "--vol-lookback-days",
        type=int,
        default=63,
        help="Lookback window for volatility targeting.",
    )
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=1.0,
        help="Maximum gross exposure after volatility targeting. 1.0 means no leverage.",
    )
    parser.add_argument(
        "--market-filter-symbol",
        default=None,
        help="Optional market regime filter symbol, e.g. SPY. Strategy is in cash when symbol is below its MA.",
    )
    parser.add_argument(
        "--market-filter-ma-days",
        type=int,
        default=200,
        help="Moving average window for the market regime filter.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional subset of symbols to include in the rotation universe.",
    )
    return parser.parse_args()


def validate_config(config: MomentumConfig, symbols: list[str]) -> None:
    if config.lookback_days <= 0:
        raise ValueError("--lookback-days must be positive.")
    if config.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if config.top_k > len(symbols):
        raise ValueError(f"--top-k={config.top_k} is larger than universe size={len(symbols)}.")
    if config.transaction_cost_bps < 0:
        raise ValueError("--transaction-cost-bps cannot be negative.")
    if config.target_volatility is not None and config.target_volatility <= 0:
        raise ValueError("--target-volatility must be positive when provided.")
    if config.vol_lookback_days <= 1:
        raise ValueError("--vol-lookback-days must be greater than 1.")
    if config.max_leverage <= 0:
        raise ValueError("--max-leverage must be positive.")
    if config.market_filter_ma_days <= 1:
        raise ValueError("--market-filter-ma-days must be greater than 1.")


def select_universe(price_matrix: pd.DataFrame, symbols: list[str] | None) -> pd.DataFrame:
    if symbols is None:
        return price_matrix

    normalized = [symbol.upper() for symbol in symbols]
    missing = sorted(set(normalized) - set(price_matrix.columns))
    if missing:
        raise ValueError(f"Requested symbols are missing from price matrix: {missing}")
    return price_matrix[normalized]


def month_end_trading_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(dates.groupby(index.to_period("M")).tail(1).values)


def build_rebalance_weights(
    price_matrix: pd.DataFrame,
    lookback_days: int,
    top_k: int,
    allow_negative_momentum: bool,
) -> pd.DataFrame:
    scores = price_matrix / price_matrix.shift(lookback_days) - 1.0
    rebalance_dates = month_end_trading_dates(price_matrix.index)
    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=price_matrix.columns)

    for rebalance_date in rebalance_dates:
        score = scores.loc[rebalance_date].dropna().sort_values(ascending=False)
        if not allow_negative_momentum:
            score = score[score > 0]
        selected = score.head(top_k).index
        if len(selected) > 0:
            weights.loc[rebalance_date, selected] = 1.0 / len(selected)

    return weights


def expand_daily_weights(rebalance_weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    weights = rebalance_weights.reindex(returns.index).ffill().fillna(0.0)
    return weights.shift(1).fillna(0.0)


def apply_market_filter(
    daily_weights: pd.DataFrame,
    full_price_matrix: pd.DataFrame,
    symbol: str | None,
    ma_days: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if symbol is None:
        regime = pd.Series(True, index=daily_weights.index, name="market_filter_active")
        return daily_weights, regime

    normalized = symbol.upper()
    if normalized not in full_price_matrix.columns:
        raise ValueError(f"Market filter symbol {normalized} is not in the price data.")

    market_price = full_price_matrix[normalized]
    moving_average = market_price.rolling(ma_days).mean()
    regime = (market_price > moving_average).shift(1).reindex(daily_weights.index).fillna(False).astype(bool)
    filtered_weights = daily_weights.mul(regime.astype(float), axis=0)
    regime.name = "market_filter_active"
    return filtered_weights, regime


def apply_volatility_target(
    daily_weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_volatility: float | None,
    vol_lookback_days: int,
    max_leverage: float,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_volatility is None:
        scale = pd.Series(1.0, index=daily_weights.index, name="exposure_scale")
        return daily_weights, scale

    raw_returns = (daily_weights * returns).sum(axis=1)
    realized_vol = raw_returns.rolling(vol_lookback_days).std() * (TRADING_DAYS**0.5)
    scale = (target_volatility / realized_vol).shift(1)
    scale = scale.clip(lower=0.0, upper=max_leverage).fillna(1.0)
    scale.name = "exposure_scale"
    return daily_weights.mul(scale, axis=0), scale


def apply_risk_overlays(
    daily_weights: pd.DataFrame,
    returns: pd.DataFrame,
    full_price_matrix: pd.DataFrame,
    config: MomentumConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_weights, regime = apply_market_filter(
        daily_weights=daily_weights,
        full_price_matrix=full_price_matrix,
        symbol=config.market_filter_symbol,
        ma_days=config.market_filter_ma_days,
    )
    scaled_weights, exposure_scale = apply_volatility_target(
        daily_weights=filtered_weights,
        returns=returns,
        target_volatility=config.target_volatility,
        vol_lookback_days=config.vol_lookback_days,
        max_leverage=config.max_leverage,
    )
    risk_state = pd.DataFrame(
        {
            "market_filter_active": regime.reindex(scaled_weights.index).astype(bool),
            "exposure_scale": exposure_scale.reindex(scaled_weights.index),
            "gross_exposure": scaled_weights.abs().sum(axis=1),
            "net_exposure": scaled_weights.sum(axis=1),
        },
        index=scaled_weights.index,
    )
    return scaled_weights, risk_state


def run_backtest(
    returns: pd.DataFrame,
    daily_weights: pd.DataFrame,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    gross_returns = (daily_weights * returns).sum(axis=1)
    turnover = daily_weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = daily_weights.iloc[0].abs().sum()
    transaction_cost = turnover * transaction_cost_bps / 10_000.0
    net_returns = gross_returns - transaction_cost

    return pd.DataFrame(
        {
            "MOMENTUM_ROTATION_GROSS": gross_returns,
            "MOMENTUM_ROTATION_NET": net_returns,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
        },
        index=returns.index,
    )


def align_to_first_position(backtest: pd.DataFrame, daily_weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    invested = daily_weights.abs().sum(axis=1) > 0
    if not invested.any():
        raise ValueError("Strategy never entered a position. Try --allow-negative-momentum or a shorter lookback.")

    first_active_date = invested[invested].index[0]
    return backtest.loc[first_active_date:], daily_weights.loc[first_active_date:]


def build_benchmark_returns(returns: pd.DataFrame) -> pd.DataFrame:
    benchmarks = pd.DataFrame(index=returns.index)
    if "SPY" in returns.columns:
        benchmarks["SPY_BUY_HOLD"] = returns["SPY"]
    if "QQQ" in returns.columns:
        benchmarks["QQQ_BUY_HOLD"] = returns["QQQ"]
    benchmarks["EW_PORTFOLIO"] = returns.mean(axis=1)
    return benchmarks


def extract_holdings(rebalance_weights: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rebalance_date, weights in rebalance_weights.iterrows():
        selected = weights[weights > 0].sort_values(ascending=False)
        for symbol, weight in selected.items():
            rows.append(
                {
                    "rebalance_date": rebalance_date.date(),
                    "symbol": symbol,
                    "weight": weight,
                }
            )
    return pd.DataFrame(rows)


def summarize_turnover(backtest: pd.DataFrame) -> pd.DataFrame:
    turnover = backtest["turnover"]
    cost = backtest["transaction_cost"]
    return pd.DataFrame(
        [
            {
                "average_daily_turnover": turnover.mean(),
                "annualized_turnover": turnover.mean() * TRADING_DAYS,
                "total_turnover": turnover.sum(),
                "total_transaction_cost": cost.sum(),
            }
        ]
    )


def summarize_risk_state(risk_state: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "average_gross_exposure": risk_state["gross_exposure"].mean(),
                "median_gross_exposure": risk_state["gross_exposure"].median(),
                "cash_days": int((risk_state["gross_exposure"] == 0).sum()),
                "cash_day_fraction": float((risk_state["gross_exposure"] == 0).mean()),
                "market_filter_active_fraction": float(risk_state["market_filter_active"].mean()),
                "average_exposure_scale": risk_state["exposure_scale"].mean(),
            }
        ]
    )


def write_equity_curve_html(wealth: pd.DataFrame, report_dir: Path) -> None:
    fig = go.Figure()
    for column in wealth.columns:
        line_width = 4 if column == "MOMENTUM_ROTATION_NET" else 2
        fig.add_trace(
            go.Scatter(
                x=wealth.index,
                y=wealth[column],
                mode="lines",
                name=column,
                line={"width": line_width},
            )
        )

    fig.update_layout(
        title="Momentum Rotation vs Benchmarks",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=720,
    )
    fig.write_html(report_dir / "equity_curves.html", include_plotlyjs="cdn")


def write_drawdown_html(wealth: pd.DataFrame, report_dir: Path) -> None:
    drawdown = wealth / wealth.cummax() - 1.0
    fig = go.Figure()
    for column in drawdown.columns:
        line_width = 4 if column == "MOMENTUM_ROTATION_NET" else 2
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown[column],
                mode="lines",
                name=column,
                line={"width": line_width},
            )
        )

    fig.update_layout(
        title="Strategy Drawdowns",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=720,
    )
    fig.write_html(report_dir / "drawdowns.html", include_plotlyjs="cdn")


def write_weights_heatmap(daily_weights: pd.DataFrame, report_dir: Path) -> None:
    monthly_weights = daily_weights.groupby(daily_weights.index.to_period("M")).tail(1)
    fig = go.Figure(
        data=go.Heatmap(
            x=monthly_weights.index,
            y=monthly_weights.columns,
            z=monthly_weights.T,
            colorscale="Viridis",
            zmin=0,
            zmax=max(1.0 / max(1, (monthly_weights > 0).sum(axis=1).max()), 1.0),
            colorbar={"title": "Weight"},
        )
    )
    fig.update_layout(
        title="Monthly Strategy Weights",
        xaxis_title="Date",
        yaxis_title="Symbol",
        template="plotly_white",
        width=1200,
        height=720,
    )
    fig.write_html(report_dir / "weights_heatmap.html", include_plotlyjs="cdn")


def write_reports(
    report_dir: Path,
    config: MomentumConfig,
    universe: list[str],
    strategy_returns: pd.DataFrame,
    comparison_returns: pd.DataFrame,
    daily_weights: pd.DataFrame,
    rebalance_weights: pd.DataFrame,
    risk_state: pd.DataFrame,
    holdings: pd.DataFrame,
    summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    risk_summary: pd.DataFrame,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    wealth = (1.0 + comparison_returns).cumprod()

    strategy_returns.to_csv(report_dir / "strategy_returns.csv", index_label="date")
    comparison_returns.to_csv(report_dir / "comparison_returns.csv", index_label="date")
    wealth.to_csv(report_dir / "cumulative_wealth.csv", index_label="date")
    daily_weights.to_csv(report_dir / "daily_weights.csv", index_label="date")
    rebalance_weights.to_csv(report_dir / "rebalance_weights.csv", index_label="date")
    risk_state.to_csv(report_dir / "risk_state.csv", index_label="date")
    holdings.to_csv(report_dir / "holdings.csv", index=False)
    summary.to_csv(report_dir / "performance_summary.csv", index=False)
    turnover_summary.to_csv(report_dir / "turnover_summary.csv", index=False)
    risk_summary.to_csv(report_dir / "risk_summary.csv", index=False)
    pd.DataFrame(
        [
            {"key": "lookback_days", "value": config.lookback_days},
            {"key": "top_k", "value": config.top_k},
            {"key": "transaction_cost_bps", "value": config.transaction_cost_bps},
            {"key": "risk_free_rate", "value": config.risk_free_rate},
            {"key": "allow_negative_momentum", "value": config.allow_negative_momentum},
            {"key": "target_volatility", "value": config.target_volatility},
            {"key": "vol_lookback_days", "value": config.vol_lookback_days},
            {"key": "max_leverage", "value": config.max_leverage},
            {"key": "market_filter_symbol", "value": config.market_filter_symbol},
            {"key": "market_filter_ma_days", "value": config.market_filter_ma_days},
            {"key": "universe", "value": ",".join(universe)},
        ]
    ).to_csv(report_dir / "strategy_config.csv", index=False)

    write_equity_curve_html(wealth, report_dir)
    write_drawdown_html(wealth, report_dir)
    write_weights_heatmap(daily_weights, report_dir)


def print_strategy_summary(summary: pd.DataFrame, turnover_summary: pd.DataFrame, risk_summary: pd.DataFrame) -> None:
    display = summary[
        [
            "symbol",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
            "calmar",
        ]
    ].copy()
    for column in ["annual_return", "annual_volatility", "max_drawdown"]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    for column in ["sharpe", "calmar"]:
        display[column] = display[column].map(lambda value: f"{value:.2f}")

    print("\nMomentum strategy comparison:")
    print(display.to_string(index=False))

    turnover = turnover_summary.iloc[0]
    print("\nTurnover:")
    print(f"annualized_turnover: {turnover['annualized_turnover']:.2f}")
    print(f"total_transaction_cost: {turnover['total_transaction_cost']:.2%}")

    risk = risk_summary.iloc[0]
    print("\nRisk overlay:")
    print(f"average_gross_exposure: {risk['average_gross_exposure']:.2f}")
    print(f"cash_day_fraction: {risk['cash_day_fraction']:.2%}")


def run_strategy(config: MomentumConfig) -> None:
    prices = load_prices(config.prices_path)
    full_price_matrix = build_price_matrix(prices)
    price_matrix = select_universe(full_price_matrix, config.symbols)
    validate_config(config, list(price_matrix.columns))

    returns = price_matrix.pct_change().dropna(how="all")
    rebalance_weights = build_rebalance_weights(
        price_matrix=price_matrix,
        lookback_days=config.lookback_days,
        top_k=config.top_k,
        allow_negative_momentum=config.allow_negative_momentum,
    )
    daily_weights = expand_daily_weights(rebalance_weights, returns)
    daily_weights, risk_state = apply_risk_overlays(
        daily_weights=daily_weights,
        returns=returns,
        full_price_matrix=full_price_matrix,
        config=config,
    )
    backtest = run_backtest(
        returns=returns,
        daily_weights=daily_weights,
        transaction_cost_bps=config.transaction_cost_bps,
    )
    backtest, daily_weights = align_to_first_position(backtest, daily_weights)
    risk_state = risk_state.loc[backtest.index]

    benchmarks = build_benchmark_returns(returns).loc[backtest.index]
    comparison_returns = pd.concat(
        [backtest[["MOMENTUM_ROTATION_NET", "MOMENTUM_ROTATION_GROSS"]], benchmarks],
        axis=1,
    )
    summary = summarize_returns(comparison_returns, config.risk_free_rate)
    turnover_summary = summarize_turnover(backtest)
    risk_summary = summarize_risk_state(risk_state)
    holdings = extract_holdings(rebalance_weights)

    write_reports(
        report_dir=config.report_dir,
        config=config,
        universe=list(price_matrix.columns),
        strategy_returns=backtest,
        comparison_returns=comparison_returns,
        daily_weights=daily_weights,
        rebalance_weights=rebalance_weights,
        risk_state=risk_state,
        holdings=holdings,
        summary=summary,
        turnover_summary=turnover_summary,
        risk_summary=risk_summary,
    )
    print_strategy_summary(summary, turnover_summary, risk_summary)

    strategy_wealth = (1.0 + comparison_returns["MOMENTUM_ROTATION_NET"]).cumprod()
    logging.info("Universe size: %d", len(price_matrix.columns))
    logging.info("Lookback days: %d, top_k: %d", config.lookback_days, config.top_k)
    logging.info("Rebalance count: %d", len(rebalance_weights))
    logging.info("Final strategy wealth: %.4f", strategy_wealth.iloc[-1])
    logging.info("Strategy max drawdown: %.2f%%", max_drawdown(strategy_wealth) * 100)
    logging.info("Wrote reports to %s", config.report_dir)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = MomentumConfig(
        prices_path=args.prices,
        report_dir=args.report_dir,
        lookback_days=args.lookback_days,
        top_k=args.top_k,
        transaction_cost_bps=args.transaction_cost_bps,
        risk_free_rate=args.risk_free_rate,
        allow_negative_momentum=args.allow_negative_momentum,
        target_volatility=args.target_volatility,
        vol_lookback_days=args.vol_lookback_days,
        max_leverage=args.max_leverage,
        market_filter_symbol=args.market_filter_symbol,
        market_filter_ma_days=args.market_filter_ma_days,
        symbols=args.symbols,
    )

    try:
        run_strategy(config)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
