from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from analyze_prices import TRADING_DAYS, build_price_matrix, load_prices, summarize_returns
from run_momentum_strategy import (
    DEFAULT_PRICES,
    align_to_first_position,
    build_benchmark_returns,
    expand_daily_weights,
    month_end_trading_dates,
    run_backtest,
    summarize_turnover,
)


DEFAULT_REPORT_DIR = Path("reports/factor_research")

FACTOR_DIRECTIONS = {
    "momentum_63": 1.0,
    "momentum_126": 1.0,
    "momentum_252": 1.0,
    "trend_200": 1.0,
    "volatility_63": -1.0,
    "volatility_126": -1.0,
    "downside_volatility_63": -1.0,
    "max_drawdown_126": 1.0,
}


@dataclass(frozen=True)
class FactorResearchConfig:
    prices_path: Path
    report_dir: Path
    forward_days: list[int]
    quantiles: int
    top_k: int
    transaction_cost_bps: float
    risk_free_rate: float
    symbols: list[str] | None


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ETF factor research and multi-factor backtest.")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--forward-days",
        default="21,63",
        help="Forward return horizons in trading days, comma-separated.",
    )
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--symbols", nargs="*", default=None)
    return parser.parse_args()


def select_universe(price_matrix: pd.DataFrame, symbols: list[str] | None) -> pd.DataFrame:
    if symbols is None:
        return price_matrix

    normalized = [symbol.upper() for symbol in symbols]
    missing = sorted(set(normalized) - set(price_matrix.columns))
    if missing:
        raise ValueError(f"Requested symbols are missing from price data: {missing}")
    return price_matrix[normalized]


def rolling_max_drawdown(price_matrix: pd.DataFrame, window: int) -> pd.DataFrame:
    def window_max_drawdown(window_prices: pd.Series) -> float:
        wealth = window_prices / window_prices.iloc[0]
        drawdown = wealth / wealth.cummax() - 1.0
        return float(drawdown.min())

    return price_matrix.rolling(window).apply(window_max_drawdown, raw=False)


def compute_factor_matrices(price_matrix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    returns = price_matrix.pct_change()
    factors = {
        "momentum_63": price_matrix / price_matrix.shift(63) - 1.0,
        "momentum_126": price_matrix / price_matrix.shift(126) - 1.0,
        "momentum_252": price_matrix / price_matrix.shift(252) - 1.0,
        "trend_200": price_matrix / price_matrix.rolling(200).mean() - 1.0,
        "volatility_63": returns.rolling(63).std() * (TRADING_DAYS**0.5),
        "volatility_126": returns.rolling(126).std() * (TRADING_DAYS**0.5),
        "downside_volatility_63": returns.where(returns < 0, 0.0).rolling(63).std() * (TRADING_DAYS**0.5),
        "max_drawdown_126": rolling_max_drawdown(price_matrix, 126),
    }
    return factors


def forward_return_matrix(price_matrix: pd.DataFrame, forward_days: int) -> pd.DataFrame:
    return price_matrix.shift(-forward_days) / price_matrix - 1.0


def build_factor_frame(
    price_matrix: pd.DataFrame,
    factor_matrices: dict[str, pd.DataFrame],
    forward_days: list[int],
) -> pd.DataFrame:
    month_ends = month_end_trading_dates(price_matrix.index)
    rows: list[pd.DataFrame] = []

    base = pd.DataFrame(index=month_ends, columns=price_matrix.columns)
    for factor_name, matrix in factor_matrices.items():
        stacked = matrix.reindex(base.index).stack().rename(factor_name)
        rows.append(stacked)

    for horizon in forward_days:
        forward_returns = forward_return_matrix(price_matrix, horizon)
        rows.append(
            forward_returns.reindex(base.index).stack().rename(f"forward_return_{horizon}")
        )

    factor_frame = pd.concat(rows, axis=1).reset_index()
    factor_frame = factor_frame.rename(columns={"level_0": "date", "level_1": "symbol"})
    factor_frame["date"] = pd.to_datetime(factor_frame["date"])
    return factor_frame


def calculate_ic_timeseries(
    factor_frame: pd.DataFrame,
    factor_names: list[str],
    forward_days: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in forward_days:
        target = f"forward_return_{horizon}"
        for factor_name in factor_names:
            for date_value, group in factor_frame.groupby("date"):
                sample = group[[factor_name, target]].dropna()
                if len(sample) < 4 or sample[factor_name].nunique() < 2 or sample[target].nunique() < 2:
                    continue
                rows.append(
                    {
                        "date": date_value,
                        "factor": factor_name,
                        "forward_days": horizon,
                        "pearson_ic": sample[factor_name].corr(sample[target], method="pearson"),
                        "rank_ic": sample[factor_name].corr(sample[target], method="spearman"),
                        "n_assets": len(sample),
                    }
                )
    return pd.DataFrame(rows)


def summarize_ic(ic_timeseries: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (factor_name, horizon), group in ic_timeseries.groupby(["factor", "forward_days"], sort=True):
        rank_ic = group["rank_ic"].dropna()
        pearson_ic = group["pearson_ic"].dropna()
        rank_std = rank_ic.std(ddof=1)
        pearson_std = pearson_ic.std(ddof=1)
        rows.append(
            {
                "factor": factor_name,
                "forward_days": horizon,
                "mean_rank_ic": rank_ic.mean(),
                "rank_ic_std": rank_std,
                "rank_ic_ir": rank_ic.mean() / rank_std if rank_std > 0 else float("nan"),
                "positive_rank_ic_fraction": (rank_ic > 0).mean(),
                "mean_pearson_ic": pearson_ic.mean(),
                "pearson_ic_ir": pearson_ic.mean() / pearson_std if pearson_std > 0 else float("nan"),
                "observations": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values(["forward_days", "mean_rank_ic"], ascending=[True, False])


def assign_quantiles(values: pd.Series, quantiles: int) -> pd.Series:
    valid = values.dropna()
    labels = pd.Series(index=values.index, dtype="float64")
    if len(valid) < quantiles or valid.nunique() < 2:
        return labels
    ranked = valid.rank(method="first")
    labels.loc[valid.index] = pd.qcut(ranked, q=quantiles, labels=False, duplicates="drop") + 1
    return labels


def calculate_quantile_returns(
    factor_frame: pd.DataFrame,
    factor_names: list[str],
    forward_days: list[int],
    quantiles: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in forward_days:
        target = f"forward_return_{horizon}"
        for factor_name in factor_names:
            working = factor_frame[["date", "symbol", factor_name, target]].dropna().copy()
            if working.empty:
                continue
            working["quantile"] = working.groupby("date")[factor_name].transform(
                lambda values: assign_quantiles(values, quantiles)
            )
            for (date_value, quantile), group in working.dropna(subset=["quantile"]).groupby(["date", "quantile"]):
                rows.append(
                    {
                        "date": date_value,
                        "factor": factor_name,
                        "forward_days": horizon,
                        "quantile": int(quantile),
                        "mean_forward_return": group[target].mean(),
                        "n_assets": len(group),
                    }
                )

    quantile_returns = pd.DataFrame(rows)
    if quantile_returns.empty:
        return quantile_returns

    summary = (
        quantile_returns.groupby(["factor", "forward_days", "quantile"], as_index=False)
        .agg(mean_forward_return=("mean_forward_return", "mean"), observations=("date", "count"))
        .sort_values(["forward_days", "factor", "quantile"])
    )
    return summary


def calculate_long_short_quantiles(quantile_returns: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    if quantile_returns.empty:
        return quantile_returns

    pivot = quantile_returns.pivot_table(
        index=["factor", "forward_days"],
        columns="quantile",
        values="mean_forward_return",
    )
    output = pd.DataFrame(
        {
            "factor": pivot.index.get_level_values("factor"),
            "forward_days": pivot.index.get_level_values("forward_days"),
            "top_quantile_return": pivot.get(quantiles),
            "bottom_quantile_return": pivot.get(1),
        }
    ).reset_index(drop=True)
    output["long_short_return"] = output["top_quantile_return"] - output["bottom_quantile_return"]
    return output.sort_values(["forward_days", "long_short_return"], ascending=[True, False])


def calculate_factor_correlation(factor_frame: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
    snapshots = []
    for _, group in factor_frame.groupby("date"):
        corr = group[factor_names].corr(method="spearman")
        snapshots.append(corr)
    return sum(snapshots) / len(snapshots)


def add_multifactor_score(
    factor_frame: pd.DataFrame,
    factor_names: list[str],
    directions: dict[str, float],
    score_column: str,
    factor_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    scored = factor_frame.copy()
    score_parts = []
    for factor_name in factor_names:
        direction = directions[factor_name]
        weight = 1.0 if factor_weights is None else factor_weights[factor_name]
        score = scored.groupby("date")[factor_name].transform(lambda values: (values * direction).rank(pct=True))
        score_parts.append((score * weight).rename(f"{score_column}_{factor_name}_score"))

    score_frame = pd.concat(score_parts, axis=1)
    scored = pd.concat([scored, score_frame], axis=1)
    if factor_weights is None:
        scored[score_column] = score_frame.mean(axis=1, skipna=False)
    else:
        denominator = pd.Series(0.0, index=score_frame.index)
        for factor_name in factor_names:
            column = f"{score_column}_{factor_name}_score"
            denominator += score_frame[column].notna().astype(float) * factor_weights[factor_name]
        scored[score_column] = score_frame.sum(axis=1, skipna=False) / denominator.replace(0.0, pd.NA)
    return scored


def build_ic_score_spec(
    ic_summary: pd.DataFrame,
    factor_names: list[str],
    forward_days: int,
) -> tuple[dict[str, float], dict[str, float]]:
    selected = ic_summary.loc[ic_summary["forward_days"] == forward_days].set_index("factor")
    directions: dict[str, float] = {}
    weights: dict[str, float] = {}
    for factor_name in factor_names:
        mean_ic = float(selected.loc[factor_name, "mean_rank_ic"]) if factor_name in selected.index else 0.0
        directions[factor_name] = 1.0 if mean_ic >= 0 else -1.0
        weights[factor_name] = max(abs(mean_ic), 1e-6)
    return directions, weights


def build_score_weights(
    scored: pd.DataFrame,
    price_matrix: pd.DataFrame,
    top_k: int,
    score_column: str,
) -> pd.DataFrame:
    rebalance_dates = month_end_trading_dates(price_matrix.index)
    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=price_matrix.columns)

    for date_value in rebalance_dates:
        score = (
            scored.loc[scored["date"] == date_value, ["symbol", score_column]]
            .dropna()
            .sort_values(score_column, ascending=False)
        )
        selected = score.head(top_k)["symbol"]
        if len(selected) > 0:
            weights.loc[date_value, selected] = 1.0 / len(selected)
    return weights


def run_multifactor_strategy(
    scored: pd.DataFrame,
    price_matrix: pd.DataFrame,
    top_k: int,
    transaction_cost_bps: float,
    risk_free_rate: float,
    score_column: str,
    return_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = price_matrix.pct_change().dropna(how="all")
    rebalance_weights = build_score_weights(scored, price_matrix, top_k, score_column)
    daily_weights = expand_daily_weights(rebalance_weights, returns)
    backtest = run_backtest(returns, daily_weights, transaction_cost_bps)
    backtest, daily_weights = align_to_first_position(backtest, daily_weights)

    benchmarks = build_benchmark_returns(returns).loc[backtest.index]
    comparison_returns = pd.concat(
        [
            backtest[["MOMENTUM_ROTATION_NET"]].rename(
                columns={"MOMENTUM_ROTATION_NET": return_column}
            ),
            benchmarks,
        ],
        axis=1,
    )
    summary = summarize_returns(comparison_returns, risk_free_rate)
    turnover = summarize_turnover(backtest)
    return comparison_returns, daily_weights, summary, turnover


def write_ic_heatmap(ic_summary: pd.DataFrame, report_dir: Path) -> None:
    pivot = ic_summary.pivot(index="factor", columns="forward_days", values="mean_rank_ic")
    fig = go.Figure(
        data=go.Heatmap(
            x=pivot.columns,
            y=pivot.index,
            z=pivot.values,
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Mean Rank IC"},
            text=pivot.round(3).astype(str).values,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Factor Mean Rank IC",
        xaxis_title="Forward Days",
        yaxis_title="Factor",
        template="plotly_white",
        width=900,
        height=650,
    )
    fig.write_html(report_dir / "ic_heatmap.html", include_plotlyjs="cdn")


def write_factor_correlation_heatmap(correlation: pd.DataFrame, report_dir: Path) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            x=correlation.columns,
            y=correlation.index,
            z=correlation.values,
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Spearman Corr"},
            text=correlation.round(2).astype(str).values,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Average Cross-Sectional Factor Correlation",
        template="plotly_white",
        width=1000,
        height=760,
    )
    fig.write_html(report_dir / "factor_correlation_heatmap.html", include_plotlyjs="cdn")


def write_multifactor_equity_curve(comparison_returns: pd.DataFrame, report_dir: Path) -> None:
    wealth = (1.0 + comparison_returns).cumprod()
    fig = go.Figure()
    for column in wealth.columns:
        line_width = 4 if column in {"MULTIFACTOR_SCORE_NET", "IC_WEIGHTED_SCORE_NET"} else 2
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
        title="Multi-Factor Score Strategy vs Benchmarks",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=720,
    )
    fig.write_html(report_dir / "multifactor_equity_curve.html", include_plotlyjs="cdn")


def format_strategy_summary(strategy_summary: pd.DataFrame) -> pd.DataFrame:
    strategy_display = strategy_summary[
        ["symbol", "annual_return", "annual_volatility", "sharpe", "max_drawdown", "calmar"]
    ].copy()
    for column in ["annual_return", "annual_volatility", "max_drawdown"]:
        strategy_display[column] = strategy_display[column].map(lambda value: f"{value:.2%}")
    for column in ["sharpe", "calmar"]:
        strategy_display[column] = strategy_display[column].map(lambda value: f"{value:.2f}")
    return strategy_display


def print_factor_summary(
    ic_summary: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    ic_strategy_summary: pd.DataFrame,
) -> None:
    print("\nTop factor IC results:")
    display = ic_summary.head(10).copy()
    for column in ["mean_rank_ic", "rank_ic_std", "rank_ic_ir", "positive_rank_ic_fraction", "mean_pearson_ic"]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    print(display.to_string(index=False))

    print("\nFixed-prior multi-factor strategy:")
    print(format_strategy_summary(strategy_summary).to_string(index=False))

    print("\nIC-weighted diagnostic strategy:")
    print(format_strategy_summary(ic_strategy_summary).to_string(index=False))


def run_research(config: FactorResearchConfig) -> None:
    if config.quantiles < 2:
        raise ValueError("--quantiles must be at least 2.")
    if config.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    prices = load_prices(config.prices_path)
    full_price_matrix = build_price_matrix(prices)
    price_matrix = select_universe(full_price_matrix, config.symbols)
    if config.top_k > len(price_matrix.columns):
        raise ValueError("--top-k cannot exceed universe size.")

    factor_matrices = compute_factor_matrices(price_matrix)
    factor_names = list(factor_matrices)
    factor_frame = build_factor_frame(price_matrix, factor_matrices, config.forward_days)
    scored = add_multifactor_score(
        factor_frame=factor_frame,
        factor_names=factor_names,
        directions=FACTOR_DIRECTIONS,
        score_column="multifactor_score",
    )
    ic_timeseries = calculate_ic_timeseries(scored, factor_names, config.forward_days)
    ic_summary = summarize_ic(ic_timeseries)
    ic_forward_days = max(config.forward_days)
    ic_directions, ic_weights = build_ic_score_spec(ic_summary, factor_names, ic_forward_days)
    scored = add_multifactor_score(
        factor_frame=scored,
        factor_names=factor_names,
        directions=ic_directions,
        score_column="ic_weighted_score",
        factor_weights=ic_weights,
    )
    quantile_returns = calculate_quantile_returns(scored, factor_names, config.forward_days, config.quantiles)
    long_short = calculate_long_short_quantiles(quantile_returns, config.quantiles)
    factor_correlation = calculate_factor_correlation(scored.dropna(subset=factor_names), factor_names)
    comparison_returns, daily_weights, strategy_summary, turnover = run_multifactor_strategy(
        scored=scored,
        price_matrix=price_matrix,
        top_k=config.top_k,
        transaction_cost_bps=config.transaction_cost_bps,
        risk_free_rate=config.risk_free_rate,
        score_column="multifactor_score",
        return_column="MULTIFACTOR_SCORE_NET",
    )
    ic_comparison_returns, ic_daily_weights, ic_strategy_summary, ic_turnover = run_multifactor_strategy(
        scored=scored,
        price_matrix=price_matrix,
        top_k=config.top_k,
        transaction_cost_bps=config.transaction_cost_bps,
        risk_free_rate=config.risk_free_rate,
        score_column="ic_weighted_score",
        return_column="IC_WEIGHTED_SCORE_NET",
    )

    config.report_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(config.report_dir / "factor_values.csv", index=False)
    ic_timeseries.to_csv(config.report_dir / "ic_timeseries.csv", index=False)
    ic_summary.to_csv(config.report_dir / "ic_summary.csv", index=False)
    quantile_returns.to_csv(config.report_dir / "quantile_returns.csv", index=False)
    long_short.to_csv(config.report_dir / "long_short_quantile_returns.csv", index=False)
    factor_correlation.to_csv(config.report_dir / "factor_correlation.csv")
    comparison_returns.to_csv(config.report_dir / "multifactor_returns.csv", index_label="date")
    daily_weights.to_csv(config.report_dir / "multifactor_daily_weights.csv", index_label="date")
    strategy_summary.to_csv(config.report_dir / "multifactor_performance_summary.csv", index=False)
    turnover.to_csv(config.report_dir / "multifactor_turnover_summary.csv", index=False)
    ic_comparison_returns.to_csv(config.report_dir / "ic_weighted_returns.csv", index_label="date")
    ic_daily_weights.to_csv(config.report_dir / "ic_weighted_daily_weights.csv", index_label="date")
    ic_strategy_summary.to_csv(config.report_dir / "ic_weighted_performance_summary.csv", index=False)
    ic_turnover.to_csv(config.report_dir / "ic_weighted_turnover_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "factor": factor_name,
                "ic_forward_days": ic_forward_days,
                "direction": ic_directions[factor_name],
                "weight": ic_weights[factor_name],
            }
            for factor_name in factor_names
        ]
    ).to_csv(config.report_dir / "ic_weighted_score_spec.csv", index=False)

    pd.DataFrame(
        [
            {"key": "forward_days", "value": ",".join(str(day) for day in config.forward_days)},
            {"key": "quantiles", "value": config.quantiles},
            {"key": "top_k", "value": config.top_k},
            {"key": "transaction_cost_bps", "value": config.transaction_cost_bps},
            {"key": "risk_free_rate", "value": config.risk_free_rate},
            {"key": "universe", "value": ",".join(price_matrix.columns)},
            {"key": "factors", "value": ",".join(factor_names)},
        ]
    ).to_csv(config.report_dir / "factor_research_config.csv", index=False)

    write_ic_heatmap(ic_summary, config.report_dir)
    write_factor_correlation_heatmap(factor_correlation, config.report_dir)
    write_multifactor_equity_curve(
        pd.concat(
            [
                comparison_returns[["MULTIFACTOR_SCORE_NET"]],
                ic_comparison_returns[["IC_WEIGHTED_SCORE_NET"]],
                comparison_returns.drop(columns=["MULTIFACTOR_SCORE_NET"]),
            ],
            axis=1,
        ),
        config.report_dir,
    )
    print_factor_summary(ic_summary, strategy_summary, ic_strategy_summary)

    logging.info("Analyzed %d factors across %d symbols", len(factor_names), len(price_matrix.columns))
    logging.info("Wrote factor research reports to %s", config.report_dir)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = FactorResearchConfig(
        prices_path=args.prices,
        report_dir=args.report_dir,
        forward_days=parse_int_list(args.forward_days),
        quantiles=args.quantiles,
        top_k=args.top_k,
        transaction_cost_bps=args.transaction_cost_bps,
        risk_free_rate=args.risk_free_rate,
        symbols=args.symbols,
    )

    try:
        run_research(config)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
