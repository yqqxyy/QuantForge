from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from analyze_prices import build_price_matrix, load_prices, summarize_returns
from run_factor_research import compute_factor_matrices, select_universe
from run_momentum_strategy import (
    DEFAULT_PRICES,
    MomentumConfig,
    align_to_first_position,
    apply_risk_overlays,
    build_benchmark_returns,
    expand_daily_weights,
    month_end_trading_dates,
    run_backtest,
    summarize_risk_state,
    summarize_turnover,
)


DEFAULT_REPORT_DIR = Path("reports/walkforward_factor_stability")


@dataclass(frozen=True)
class WalkForwardConfig:
    prices_path: Path
    report_dir: Path
    forward_days: int
    train_months: int
    min_train_months: int
    min_effective_months: int
    min_abs_ic: float
    min_directional_ic_fraction: float
    min_abs_ic_ir: float
    decorrelate_factors: bool
    factor_corr_threshold: float
    top_k: int
    transaction_cost_bps: float
    risk_free_rate: float
    target_volatility: float | None
    vol_lookback_days: int
    max_leverage: float
    market_filter_symbol: str | None
    market_filter_ma_days: int
    symbols: list[str] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward IC-weighted factor strategy.")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--forward-days", type=int, default=63)
    parser.add_argument("--train-months", type=int, default=36)
    parser.add_argument("--min-train-months", type=int, default=18)
    parser.add_argument(
        "--min-effective-months",
        type=int,
        default=12,
        help="Minimum months with valid IC observations required for a factor to be eligible.",
    )
    parser.add_argument(
        "--min-abs-ic",
        type=float,
        default=0.03,
        help="Minimum absolute rolling mean Rank IC required for a factor.",
    )
    parser.add_argument(
        "--min-directional-ic-fraction",
        type=float,
        default=0.55,
        help="Minimum fraction of IC observations matching the factor's estimated direction.",
    )
    parser.add_argument(
        "--min-abs-ic-ir",
        type=float,
        default=0.10,
        help="Minimum absolute IC information ratio required for a factor.",
    )
    parser.add_argument(
        "--decorrelate-factors",
        action="store_true",
        help="Cluster correlated selected factors and keep one representative from each cluster.",
    )
    parser.add_argument(
        "--factor-corr-threshold",
        type=float,
        default=0.80,
        help="Absolute average Spearman correlation threshold used for factor clustering.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--target-volatility", type=float, default=None)
    parser.add_argument("--vol-lookback-days", type=int, default=63)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--market-filter-symbol", default=None)
    parser.add_argument("--market-filter-ma-days", type=int, default=200)
    parser.add_argument("--symbols", nargs="*", default=None)
    return parser.parse_args()


def validate_config(config: WalkForwardConfig, universe_size: int) -> None:
    if config.forward_days <= 0:
        raise ValueError("--forward-days must be positive.")
    if config.train_months <= 0:
        raise ValueError("--train-months must be positive.")
    if config.min_train_months <= 0:
        raise ValueError("--min-train-months must be positive.")
    if config.min_train_months > config.train_months:
        raise ValueError("--min-train-months cannot exceed --train-months.")
    if config.min_effective_months <= 0:
        raise ValueError("--min-effective-months must be positive.")
    if config.min_effective_months > config.train_months:
        raise ValueError("--min-effective-months cannot exceed --train-months.")
    if config.min_abs_ic < 0:
        raise ValueError("--min-abs-ic cannot be negative.")
    if not 0 <= config.min_directional_ic_fraction <= 1:
        raise ValueError("--min-directional-ic-fraction must be between 0 and 1.")
    if config.min_abs_ic_ir < 0:
        raise ValueError("--min-abs-ic-ir cannot be negative.")
    if not 0 <= config.factor_corr_threshold <= 1:
        raise ValueError("--factor-corr-threshold must be between 0 and 1.")
    if config.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if config.top_k > universe_size:
        raise ValueError("--top-k cannot exceed universe size.")
    if config.transaction_cost_bps < 0:
        raise ValueError("--transaction-cost-bps cannot be negative.")
    if config.target_volatility is not None and config.target_volatility <= 0:
        raise ValueError("--target-volatility must be positive when provided.")


def month_end_factor_frame(
    price_matrix: pd.DataFrame,
    factor_matrices: dict[str, pd.DataFrame],
    forward_days: int,
) -> pd.DataFrame:
    month_ends = month_end_trading_dates(price_matrix.index)
    rows: list[dict[str, object]] = []
    forward_returns = price_matrix.shift(-forward_days) / price_matrix - 1.0

    for date_value in month_ends:
        for symbol in price_matrix.columns:
            row: dict[str, object] = {"date": date_value, "symbol": symbol}
            for factor_name, matrix in factor_matrices.items():
                row[factor_name] = matrix.at[date_value, symbol]
            row["forward_return"] = forward_returns.at[date_value, symbol]
            rows.append(row)
    return pd.DataFrame(rows)


def realized_training_cutoff(price_index: pd.DatetimeIndex, date_value: pd.Timestamp, forward_days: int) -> pd.Timestamp | None:
    location = price_index.get_loc(date_value)
    if not isinstance(location, int):
        location = int(location.start)
    cutoff_location = location - forward_days
    if cutoff_location < 0:
        return None
    return price_index[cutoff_location]


def estimate_factor_stability(
    factor_frame: pd.DataFrame,
    factor_names: list[str],
    train_dates: pd.DatetimeIndex,
    config: WalkForwardConfig,
) -> pd.DataFrame:
    training = factor_frame.loc[factor_frame["date"].isin(train_dates)]
    rows: list[dict[str, object]] = []
    for factor_name in factor_names:
        ics = []
        for _, group in training.groupby("date"):
            sample = group[[factor_name, "forward_return"]].dropna()
            if len(sample) < 4 or sample[factor_name].nunique() < 2 or sample["forward_return"].nunique() < 2:
                continue
            ics.append(sample[factor_name].corr(sample["forward_return"], method="spearman"))

        ic_series = pd.Series(ics, dtype="float64").dropna()
        mean_ic = float(ic_series.mean()) if not ic_series.empty else float("nan")
        ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else float("nan")
        ic_ir = mean_ic / ic_std if pd.notna(ic_std) and ic_std > 0 else float("nan")
        direction = 1.0 if pd.notna(mean_ic) and mean_ic >= 0 else -1.0
        directional_fraction = float((ic_series * direction > 0).mean()) if not ic_series.empty else 0.0
        selected = (
            len(ic_series) >= config.min_effective_months
            and pd.notna(mean_ic)
            and abs(mean_ic) >= config.min_abs_ic
            and directional_fraction >= config.min_directional_ic_fraction
            and pd.notna(ic_ir)
            and abs(ic_ir) >= config.min_abs_ic_ir
        )

        rows.append(
            {
                "factor": factor_name,
                "mean_rank_ic": mean_ic,
                "rank_ic_std": ic_std,
                "rank_ic_ir": ic_ir,
                "direction": direction,
                "directional_ic_fraction": directional_fraction,
                "effective_months": len(ic_series),
                "selected": selected,
                "weight": abs(mean_ic) if selected and pd.notna(mean_ic) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def average_factor_rank_correlation(
    factor_frame: pd.DataFrame,
    factor_names: list[str],
    train_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    training = factor_frame.loc[factor_frame["date"].isin(train_dates)]
    snapshots = []
    for _, group in training.groupby("date"):
        ranks = group[factor_names].rank(pct=True)
        corr = ranks.corr(method="spearman")
        if not corr.isna().all().all():
            snapshots.append(corr)

    if not snapshots:
        return pd.DataFrame(0.0, index=factor_names, columns=factor_names)

    average = sum(snapshots) / len(snapshots)
    return average.reindex(index=factor_names, columns=factor_names).fillna(0.0)


def connected_factor_clusters(
    factors: list[str],
    correlation: pd.DataFrame,
    threshold: float,
) -> list[list[str]]:
    remaining = set(factors)
    clusters: list[list[str]] = []

    while remaining:
        start = remaining.pop()
        stack = [start]
        cluster = {start}
        while stack:
            current = stack.pop()
            for other in list(remaining):
                if abs(float(correlation.loc[current, other])) >= threshold:
                    remaining.remove(other)
                    cluster.add(other)
                    stack.append(other)
        clusters.append(sorted(cluster))

    return sorted(clusters, key=lambda items: (len(items), items[0]), reverse=True)


def apply_factor_decorrelation(
    factor_stability: pd.DataFrame,
    factor_correlation: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    output = factor_stability.copy()
    output["selected_before_cluster"] = output["selected"].astype(bool)
    output["cluster_id"] = pd.NA
    output["cluster_size"] = 0
    output["cluster_representative"] = False
    output["suppressed_by_cluster"] = False

    selected_factors = output.loc[output["selected_before_cluster"], "factor"].tolist()
    if not selected_factors:
        return output

    clusters = connected_factor_clusters(selected_factors, factor_correlation, threshold)
    for cluster_index, cluster in enumerate(clusters, start=1):
        cluster_id = f"C{cluster_index:02d}"
        cluster_frame = output.loc[output["factor"].isin(cluster)].copy()
        cluster_frame["abs_ic_ir"] = cluster_frame["rank_ic_ir"].abs()
        representative = (
            cluster_frame.sort_values(
                ["weight", "abs_ic_ir", "directional_ic_fraction", "factor"],
                ascending=[False, False, False, True],
            )
            .iloc[0]["factor"]
        )

        cluster_mask = output["factor"].isin(cluster)
        output.loc[cluster_mask, "cluster_id"] = cluster_id
        output.loc[cluster_mask, "cluster_size"] = len(cluster)
        output.loc[output["factor"] == representative, "cluster_representative"] = True
        suppressed = cluster_mask & (output["factor"] != representative)
        output.loc[suppressed, "selected"] = False
        output.loc[suppressed, "weight"] = 0.0
        output.loc[suppressed, "suppressed_by_cluster"] = True

    return output


def score_rebalance_date(
    factor_frame: pd.DataFrame,
    factor_names: list[str],
    rebalance_date: pd.Timestamp,
    factor_stability: pd.DataFrame,
) -> pd.Series:
    current = factor_frame.loc[factor_frame["date"] == rebalance_date, ["symbol", *factor_names]].copy()
    score = pd.Series(0.0, index=current.index)
    denominator = pd.Series(0.0, index=current.index)
    selected_factors = factor_stability.loc[factor_stability["selected"]].set_index("factor")

    for factor_name, row in selected_factors.iterrows():
        direction = float(row["direction"])
        weight = float(row["weight"])
        if weight <= 0:
            continue
        ranks = (current[factor_name] * direction).rank(pct=True)
        score = score.add(ranks.fillna(0.0) * weight, fill_value=0.0)
        denominator = denominator.add(ranks.notna().astype(float) * weight, fill_value=0.0)

    output = pd.Series(index=current["symbol"], dtype="float64")
    valid = denominator > 0
    output.loc[current.loc[valid, "symbol"]] = (score.loc[valid] / denominator.loc[valid]).values
    output.name = "walkforward_score"
    return output


def build_walkforward_weights(
    price_matrix: pd.DataFrame,
    factor_frame: pd.DataFrame,
    factor_names: list[str],
    config: WalkForwardConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rebalance_dates = month_end_trading_dates(price_matrix.index)
    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=price_matrix.columns)
    ic_rows: list[dict[str, object]] = []

    for rebalance_date in rebalance_dates:
        cutoff_date = realized_training_cutoff(price_matrix.index, rebalance_date, config.forward_days)
        if cutoff_date is None:
            continue

        eligible_dates = rebalance_dates[rebalance_dates <= cutoff_date]
        train_dates = eligible_dates[-config.train_months :]
        if len(train_dates) < config.min_train_months:
            continue

        factor_stability = estimate_factor_stability(factor_frame, factor_names, train_dates, config)
        factor_correlation = average_factor_rank_correlation(factor_frame, factor_names, train_dates)
        if config.decorrelate_factors:
            factor_stability = apply_factor_decorrelation(
                factor_stability=factor_stability,
                factor_correlation=factor_correlation,
                threshold=config.factor_corr_threshold,
            )
        else:
            factor_stability["selected_before_cluster"] = factor_stability["selected"].astype(bool)
            factor_stability["cluster_id"] = pd.NA
            factor_stability["cluster_size"] = 0
            factor_stability["cluster_representative"] = False
            factor_stability["suppressed_by_cluster"] = False
        scores = score_rebalance_date(factor_frame, factor_names, rebalance_date, factor_stability)
        selected = scores.dropna().sort_values(ascending=False).head(config.top_k)
        if len(selected) > 0:
            weights.loc[rebalance_date, selected.index] = 1.0 / len(selected)

        for row in factor_stability.itertuples(index=False):
            ic_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "factor": row.factor,
                    "mean_rank_ic": row.mean_rank_ic,
                    "rank_ic_std": row.rank_ic_std,
                    "rank_ic_ir": row.rank_ic_ir,
                    "direction": row.direction,
                    "directional_ic_fraction": row.directional_ic_fraction,
                    "effective_months": row.effective_months,
                    "selected_before_cluster": row.selected_before_cluster,
                    "cluster_id": row.cluster_id,
                    "cluster_size": row.cluster_size,
                    "cluster_representative": row.cluster_representative,
                    "suppressed_by_cluster": row.suppressed_by_cluster,
                    "selected": row.selected,
                    "weight": row.weight,
                }
            )

    return weights, pd.DataFrame(ic_rows)


def walkforward_risk_config(config: WalkForwardConfig) -> MomentumConfig:
    return MomentumConfig(
        prices_path=config.prices_path,
        report_dir=config.report_dir,
        lookback_days=252,
        top_k=config.top_k,
        transaction_cost_bps=config.transaction_cost_bps,
        risk_free_rate=config.risk_free_rate,
        allow_negative_momentum=True,
        target_volatility=config.target_volatility,
        vol_lookback_days=config.vol_lookback_days,
        max_leverage=config.max_leverage,
        market_filter_symbol=config.market_filter_symbol,
        market_filter_ma_days=config.market_filter_ma_days,
        symbols=config.symbols,
    )


def extract_holdings(weights: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rebalance_date, row in weights.iterrows():
        selected = row[row > 0].sort_values(ascending=False)
        for symbol, weight in selected.items():
            rows.append({"rebalance_date": rebalance_date.date(), "symbol": symbol, "weight": weight})
    return pd.DataFrame(rows)


def write_equity_curve(comparison_returns: pd.DataFrame, report_dir: Path) -> None:
    wealth = (1.0 + comparison_returns).cumprod()
    fig = go.Figure()
    for column in wealth.columns:
        line_width = 4 if column == "WALKFORWARD_FACTOR_NET" else 2
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
        title="Walk-Forward Factor Strategy vs Benchmarks",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=720,
    )
    fig.write_html(report_dir / "equity_curve.html", include_plotlyjs="cdn")


def write_factor_ic_chart(ic_history: pd.DataFrame, report_dir: Path) -> None:
    fig = go.Figure()
    for factor_name, group in ic_history.groupby("factor"):
        fig.add_trace(
            go.Scatter(
                x=group["rebalance_date"],
                y=group["mean_rank_ic"],
                mode="lines",
                name=factor_name,
            )
        )
    fig.update_layout(
        title="Walk-Forward Rolling Mean Rank IC",
        xaxis_title="Rebalance Date",
        yaxis_title="Mean Rank IC",
        hovermode="x unified",
        template="plotly_white",
        width=1200,
        height=720,
    )
    fig.write_html(report_dir / "rolling_ic.html", include_plotlyjs="cdn")


def summarize_factor_selection(ic_history: pd.DataFrame) -> pd.DataFrame:
    if ic_history.empty:
        return pd.DataFrame()

    summary = (
        ic_history.groupby("factor", as_index=False)
        .agg(
            selection_rate=("selected", "mean"),
            pre_cluster_selection_rate=("selected_before_cluster", "mean"),
            cluster_suppression_rate=("suppressed_by_cluster", "mean"),
            representative_rate=("cluster_representative", "mean"),
            average_weight=("weight", "mean"),
            average_abs_ic=("mean_rank_ic", lambda values: values.abs().mean()),
            average_abs_ic_ir=("rank_ic_ir", lambda values: values.abs().mean()),
            average_directional_fraction=("directional_ic_fraction", "mean"),
            last_selected=("selected", "last"),
            last_mean_rank_ic=("mean_rank_ic", "last"),
            last_weight=("weight", "last"),
        )
        .sort_values(["selection_rate", "average_weight"], ascending=False)
    )
    return summary


def write_factor_selection_heatmap(ic_history: pd.DataFrame, report_dir: Path) -> None:
    if ic_history.empty:
        return

    selection = ic_history.pivot_table(
        index="factor",
        columns="rebalance_date",
        values="selected",
        aggfunc="max",
        fill_value=False,
    ).astype(int)
    fig = go.Figure(
        data=go.Heatmap(
            x=selection.columns,
            y=selection.index,
            z=selection.values,
            colorscale=[[0, "#f1f5f9"], [1, "#0f766e"]],
            zmin=0,
            zmax=1,
            colorbar={"title": "Selected"},
        )
    )
    fig.update_layout(
        title="Factor Stability Selection",
        xaxis_title="Rebalance Date",
        yaxis_title="Factor",
        template="plotly_white",
        width=1200,
        height=640,
    )
    fig.write_html(report_dir / "factor_selection_heatmap.html", include_plotlyjs="cdn")


def write_cluster_representative_heatmap(ic_history: pd.DataFrame, report_dir: Path) -> None:
    if ic_history.empty or "cluster_representative" not in ic_history.columns:
        return

    representative = ic_history.pivot_table(
        index="factor",
        columns="rebalance_date",
        values="cluster_representative",
        aggfunc="max",
        fill_value=False,
    ).astype(int)
    fig = go.Figure(
        data=go.Heatmap(
            x=representative.columns,
            y=representative.index,
            z=representative.values,
            colorscale=[[0, "#f8fafc"], [1, "#7c3aed"]],
            zmin=0,
            zmax=1,
            colorbar={"title": "Cluster Rep"},
        )
    )
    fig.update_layout(
        title="Factor Cluster Representatives",
        xaxis_title="Rebalance Date",
        yaxis_title="Factor",
        template="plotly_white",
        width=1200,
        height=640,
    )
    fig.write_html(report_dir / "cluster_representative_heatmap.html", include_plotlyjs="cdn")


def write_weight_heatmap(daily_weights: pd.DataFrame, report_dir: Path) -> None:
    monthly_weights = daily_weights.groupby(daily_weights.index.to_period("M")).tail(1)
    fig = go.Figure(
        data=go.Heatmap(
            x=monthly_weights.index,
            y=monthly_weights.columns,
            z=monthly_weights.T,
            colorscale="Viridis",
            zmin=0,
            zmax=1.0,
            colorbar={"title": "Weight"},
        )
    )
    fig.update_layout(
        title="Walk-Forward Monthly Weights",
        xaxis_title="Date",
        yaxis_title="Symbol",
        template="plotly_white",
        width=1200,
        height=720,
    )
    fig.write_html(report_dir / "weights_heatmap.html", include_plotlyjs="cdn")


def print_summary(summary: pd.DataFrame, risk_summary: pd.DataFrame, factor_selection_summary: pd.DataFrame) -> None:
    display = summary[["symbol", "annual_return", "annual_volatility", "sharpe", "max_drawdown", "calmar"]].copy()
    for column in ["annual_return", "annual_volatility", "max_drawdown"]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    for column in ["sharpe", "calmar"]:
        display[column] = display[column].map(lambda value: f"{value:.2f}")

    print("\nWalk-forward factor strategy:")
    print(display.to_string(index=False))
    risk = risk_summary.iloc[0]
    print("\nRisk state:")
    print(f"average_gross_exposure: {risk['average_gross_exposure']:.2f}")
    print(f"cash_day_fraction: {risk['cash_day_fraction']:.2%}")

    if not factor_selection_summary.empty:
        factor_display = factor_selection_summary[
            [
                "factor",
                "selection_rate",
                "pre_cluster_selection_rate",
                "cluster_suppression_rate",
                "average_abs_ic",
                "last_selected",
            ]
        ].copy()
        factor_display["selection_rate"] = factor_display["selection_rate"].map(lambda value: f"{value:.2%}")
        factor_display["pre_cluster_selection_rate"] = factor_display["pre_cluster_selection_rate"].map(lambda value: f"{value:.2%}")
        factor_display["cluster_suppression_rate"] = factor_display["cluster_suppression_rate"].map(lambda value: f"{value:.2%}")
        factor_display["average_abs_ic"] = factor_display["average_abs_ic"].map(lambda value: f"{value:.3f}")
        print("\nMost stable factors:")
        print(factor_display.head(8).to_string(index=False))


def run_walkforward(config: WalkForwardConfig) -> None:
    prices = load_prices(config.prices_path)
    full_price_matrix = build_price_matrix(prices)
    price_matrix = select_universe(full_price_matrix, config.symbols)
    validate_config(config, len(price_matrix.columns))

    factor_matrices = compute_factor_matrices(price_matrix)
    factor_names = list(factor_matrices)
    factor_frame = month_end_factor_frame(price_matrix, factor_matrices, config.forward_days)
    rebalance_weights, ic_history = build_walkforward_weights(price_matrix, factor_frame, factor_names, config)

    returns = price_matrix.pct_change().dropna(how="all")
    daily_weights = expand_daily_weights(rebalance_weights, returns)
    daily_weights, risk_state = apply_risk_overlays(
        daily_weights=daily_weights,
        returns=returns,
        full_price_matrix=full_price_matrix,
        config=walkforward_risk_config(config),
    )
    backtest = run_backtest(returns, daily_weights, config.transaction_cost_bps)
    backtest, daily_weights = align_to_first_position(backtest, daily_weights)
    risk_state = risk_state.loc[backtest.index]

    strategy_returns = backtest[["MOMENTUM_ROTATION_NET"]].rename(
        columns={"MOMENTUM_ROTATION_NET": "WALKFORWARD_FACTOR_NET"}
    )
    benchmarks = build_benchmark_returns(returns).loc[backtest.index]
    comparison_returns = pd.concat([strategy_returns, benchmarks], axis=1)
    summary = summarize_returns(comparison_returns, config.risk_free_rate)
    turnover = summarize_turnover(backtest)
    risk_summary = summarize_risk_state(risk_state)
    holdings = extract_holdings(rebalance_weights)
    factor_selection_summary = summarize_factor_selection(ic_history)

    config.report_dir.mkdir(parents=True, exist_ok=True)
    factor_frame.to_csv(config.report_dir / "factor_values_with_forward_returns.csv", index=False)
    rebalance_weights.to_csv(config.report_dir / "rebalance_weights.csv", index_label="date")
    daily_weights.to_csv(config.report_dir / "daily_weights.csv", index_label="date")
    holdings.to_csv(config.report_dir / "holdings.csv", index=False)
    ic_history.to_csv(config.report_dir / "rolling_ic_weights.csv", index=False)
    factor_selection_summary.to_csv(config.report_dir / "factor_selection_summary.csv", index=False)
    backtest.to_csv(config.report_dir / "strategy_returns.csv", index_label="date")
    comparison_returns.to_csv(config.report_dir / "comparison_returns.csv", index_label="date")
    (1.0 + comparison_returns).cumprod().to_csv(config.report_dir / "cumulative_wealth.csv", index_label="date")
    risk_state.to_csv(config.report_dir / "risk_state.csv", index_label="date")
    summary.to_csv(config.report_dir / "performance_summary.csv", index=False)
    turnover.to_csv(config.report_dir / "turnover_summary.csv", index=False)
    risk_summary.to_csv(config.report_dir / "risk_summary.csv", index=False)
    pd.DataFrame(
        [
            {"key": "forward_days", "value": config.forward_days},
            {"key": "train_months", "value": config.train_months},
            {"key": "min_train_months", "value": config.min_train_months},
            {"key": "min_effective_months", "value": config.min_effective_months},
            {"key": "min_abs_ic", "value": config.min_abs_ic},
            {"key": "min_directional_ic_fraction", "value": config.min_directional_ic_fraction},
            {"key": "min_abs_ic_ir", "value": config.min_abs_ic_ir},
            {"key": "decorrelate_factors", "value": config.decorrelate_factors},
            {"key": "factor_corr_threshold", "value": config.factor_corr_threshold},
            {"key": "top_k", "value": config.top_k},
            {"key": "transaction_cost_bps", "value": config.transaction_cost_bps},
            {"key": "target_volatility", "value": config.target_volatility},
            {"key": "market_filter_symbol", "value": config.market_filter_symbol},
            {"key": "universe", "value": ",".join(price_matrix.columns)},
            {"key": "factors", "value": ",".join(factor_names)},
        ]
    ).to_csv(config.report_dir / "walkforward_config.csv", index=False)

    write_equity_curve(comparison_returns, config.report_dir)
    write_factor_ic_chart(ic_history, config.report_dir)
    write_factor_selection_heatmap(ic_history, config.report_dir)
    write_cluster_representative_heatmap(ic_history, config.report_dir)
    write_weight_heatmap(daily_weights, config.report_dir)
    print_summary(summary, risk_summary, factor_selection_summary)

    logging.info("Walk-forward rebalances with IC estimates: %d", ic_history["rebalance_date"].nunique())
    logging.info("Wrote walk-forward reports to %s", config.report_dir)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = WalkForwardConfig(
        prices_path=args.prices,
        report_dir=args.report_dir,
        forward_days=args.forward_days,
        train_months=args.train_months,
        min_train_months=args.min_train_months,
        min_effective_months=args.min_effective_months,
        min_abs_ic=args.min_abs_ic,
        min_directional_ic_fraction=args.min_directional_ic_fraction,
        min_abs_ic_ir=args.min_abs_ic_ir,
        decorrelate_factors=args.decorrelate_factors,
        factor_corr_threshold=args.factor_corr_threshold,
        top_k=args.top_k,
        transaction_cost_bps=args.transaction_cost_bps,
        risk_free_rate=args.risk_free_rate,
        target_volatility=args.target_volatility,
        vol_lookback_days=args.vol_lookback_days,
        max_leverage=args.max_leverage,
        market_filter_symbol=args.market_filter_symbol,
        market_filter_ma_days=args.market_filter_ma_days,
        symbols=args.symbols,
    )

    try:
        run_walkforward(config)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
