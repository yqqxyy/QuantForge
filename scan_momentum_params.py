from __future__ import annotations

import argparse
import itertools
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from analyze_prices import build_price_matrix, load_prices, summarize_returns
from run_momentum_strategy import (
    DEFAULT_PRICES,
    MomentumConfig,
    align_to_first_position,
    apply_risk_overlays,
    build_rebalance_weights,
    expand_daily_weights,
    run_backtest,
    select_universe,
    summarize_risk_state,
    summarize_turnover,
    validate_config,
)


DEFAULT_REPORT_DIR = Path("reports/parameter_scan")


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_optional_float_list(value: str) -> list[float | None]:
    parsed: list[float | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        parsed.append(None if item in {"none", "null", "cash"} else float(item))
    return parsed


def parse_market_filters(value: str) -> list[str | None]:
    parsed: list[str | None] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(None if item.lower() in {"none", "null"} else item.upper())
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan momentum strategy parameters and risk overlays.")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--lookbacks", default="63,126,189,252")
    parser.add_argument("--top-ks", default="1,2,3,5")
    parser.add_argument("--target-volatilities", default="none,0.12,0.15")
    parser.add_argument("--market-filters", default="none,SPY")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--vol-lookback-days", type=int, default=63)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--market-filter-ma-days", type=int, default=200)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--allow-negative-momentum", action="store_true")
    parser.add_argument("--symbols", nargs="*", default=None)
    return parser.parse_args()


def base_config(args: argparse.Namespace) -> MomentumConfig:
    return MomentumConfig(
        prices_path=args.prices,
        report_dir=args.report_dir,
        lookback_days=252,
        top_k=3,
        transaction_cost_bps=args.transaction_cost_bps,
        risk_free_rate=args.risk_free_rate,
        allow_negative_momentum=args.allow_negative_momentum,
        target_volatility=None,
        vol_lookback_days=args.vol_lookback_days,
        max_leverage=args.max_leverage,
        market_filter_symbol=None,
        market_filter_ma_days=args.market_filter_ma_days,
        symbols=args.symbols,
    )


def evaluate_config(
    config: MomentumConfig,
    full_price_matrix: pd.DataFrame,
    price_matrix: pd.DataFrame,
) -> dict[str, object]:
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

    summary = summarize_returns(backtest[["MOMENTUM_ROTATION_NET"]], config.risk_free_rate)
    metrics = summary.iloc[0].to_dict()
    turnover = summarize_turnover(backtest).iloc[0].to_dict()
    risk = summarize_risk_state(risk_state).iloc[0].to_dict()

    return {
        "lookback_days": config.lookback_days,
        "top_k": config.top_k,
        "target_volatility": config.target_volatility,
        "market_filter_symbol": config.market_filter_symbol,
        "market_filter_ma_days": config.market_filter_ma_days if config.market_filter_symbol else None,
        "vol_lookback_days": config.vol_lookback_days if config.target_volatility else None,
        "max_leverage": config.max_leverage if config.target_volatility else None,
        "observations": int(metrics["observations"]),
        "total_return": metrics["total_return"],
        "annual_return": metrics["annual_return"],
        "annual_volatility": metrics["annual_volatility"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "calmar": metrics["calmar"],
        "annualized_turnover": turnover["annualized_turnover"],
        "total_transaction_cost": turnover["total_transaction_cost"],
        "average_gross_exposure": risk["average_gross_exposure"],
        "cash_day_fraction": risk["cash_day_fraction"],
        "market_filter_active_fraction": risk["market_filter_active_fraction"],
    }


def write_sharpe_heatmap(results: pd.DataFrame, report_dir: Path) -> None:
    best = results.groupby(["lookback_days", "top_k"], as_index=False)["sharpe"].max()
    pivot = best.pivot(index="lookback_days", columns="top_k", values="sharpe").sort_index()
    fig = go.Figure(
        data=go.Heatmap(
            x=pivot.columns,
            y=pivot.index,
            z=pivot.values,
            colorscale="Viridis",
            colorbar={"title": "Sharpe"},
            text=pivot.round(2).astype(str).values,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Best Sharpe by Lookback and Top-K",
        xaxis_title="Top-K",
        yaxis_title="Lookback Days",
        template="plotly_white",
        width=900,
        height=650,
    )
    fig.write_html(report_dir / "best_sharpe_heatmap.html", include_plotlyjs="cdn")


def write_risk_return_scatter(results: pd.DataFrame, report_dir: Path) -> None:
    fig = go.Figure()
    labels = results["market_filter_symbol"].fillna("none") + " | vol=" + results["target_volatility"].fillna("none").astype(str)
    for label in sorted(labels.unique()):
        mask = labels == label
        frame = results.loc[mask]
        fig.add_trace(
            go.Scatter(
                x=frame["annual_volatility"],
                y=frame["annual_return"],
                mode="markers",
                name=label,
                marker={
                    "size": 8 + 30 * frame["max_drawdown"].abs(),
                    "opacity": 0.75,
                },
                text=[
                    f"lookback={row.lookback_days}, top_k={row.top_k}, sharpe={row.sharpe:.2f}, mdd={row.max_drawdown:.2%}"
                    for row in frame.itertuples()
                ],
                hovertemplate="%{text}<br>ann_return=%{y:.2%}<br>ann_vol=%{x:.2%}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Momentum Parameter Scan: Return vs Volatility",
        xaxis_title="Annual Volatility",
        yaxis_title="Annual Return",
        template="plotly_white",
        width=1100,
        height=700,
    )
    fig.write_html(report_dir / "risk_return_scatter.html", include_plotlyjs="cdn")


def write_best_command(best: pd.Series, args: argparse.Namespace, report_dir: Path) -> None:
    parts = [
        "python run_momentum_strategy.py",
        f"--lookback-days {int(best['lookback_days'])}",
        f"--top-k {int(best['top_k'])}",
        f"--transaction-cost-bps {args.transaction_cost_bps}",
        "--report-dir reports/momentum_rotation_best",
    ]
    if pd.notna(best["target_volatility"]):
        parts.extend(
            [
                f"--target-volatility {best['target_volatility']}",
                f"--vol-lookback-days {args.vol_lookback_days}",
                f"--max-leverage {args.max_leverage}",
            ]
        )
    if pd.notna(best["market_filter_symbol"]):
        parts.extend(
            [
                f"--market-filter-symbol {best['market_filter_symbol']}",
                f"--market-filter-ma-days {args.market_filter_ma_days}",
            ]
        )
    if args.allow_negative_momentum:
        parts.append("--allow-negative-momentum")
    if args.symbols:
        parts.append("--symbols " + " ".join(args.symbols))

    (report_dir / "best_config_command.txt").write_text(" ".join(parts) + "\n", encoding="utf-8")


def run_scan(args: argparse.Namespace) -> pd.DataFrame:
    full_price_matrix = build_price_matrix(load_prices(args.prices))
    price_matrix = select_universe(full_price_matrix, args.symbols)
    config = base_config(args)

    lookbacks = parse_int_list(args.lookbacks)
    top_ks = parse_int_list(args.top_ks)
    target_volatilities = parse_optional_float_list(args.target_volatilities)
    market_filters = parse_market_filters(args.market_filters)

    rows: list[dict[str, object]] = []
    grid = itertools.product(lookbacks, top_ks, target_volatilities, market_filters)
    for lookback_days, top_k, target_volatility, market_filter_symbol in grid:
        trial = replace(
            config,
            lookback_days=lookback_days,
            top_k=top_k,
            target_volatility=target_volatility,
            market_filter_symbol=market_filter_symbol,
        )
        try:
            rows.append(evaluate_config(trial, full_price_matrix, price_matrix))
        except Exception as exc:
            logging.warning(
                "Skipped lookback=%s top_k=%s target_vol=%s market_filter=%s: %s",
                lookback_days,
                top_k,
                target_volatility,
                market_filter_symbol,
                exc,
            )

    if not rows:
        raise ValueError("No scan results were produced.")
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def print_scan_summary(results: pd.DataFrame) -> None:
    display = results.head(10)[
        [
            "lookback_days",
            "top_k",
            "target_volatility",
            "market_filter_symbol",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
            "cash_day_fraction",
        ]
    ].copy()
    for column in ["annual_return", "annual_volatility", "max_drawdown", "cash_day_fraction"]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    display["sharpe"] = display["sharpe"].map(lambda value: f"{value:.2f}")
    print("\nTop parameter sets by Sharpe:")
    print(display.to_string(index=False))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    try:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        results = run_scan(args)
        best_by_sharpe = results.head(20)
        best_by_calmar = results.sort_values("calmar", ascending=False).head(20)

        results.to_csv(args.report_dir / "momentum_scan_results.csv", index=False)
        best_by_sharpe.to_csv(args.report_dir / "best_by_sharpe.csv", index=False)
        best_by_calmar.to_csv(args.report_dir / "best_by_calmar.csv", index=False)
        write_sharpe_heatmap(results, args.report_dir)
        write_risk_return_scatter(results, args.report_dir)
        write_best_command(results.iloc[0], args, args.report_dir)
        print_scan_summary(results)
        logging.info("Wrote scan reports to %s", args.report_dir)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
