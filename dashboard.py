# conda run -n quantforge streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


REPORTS_DIR = Path("reports")
DEFAULT_REPORT = Path("reports/walkforward_factor_clustered_loose")
MARKET_REGIME_REPORT = Path("reports/market_regime")
AI_CHIP_REPORT = Path("reports/ai_chip_trends")
STRATEGY_SYMBOL = "WALKFORWARD_FACTOR_NET"


st.set_page_config(
    page_title="QuantForge Dashboard",
    page_icon="QF",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
      :root {
        --qf-border: #d7dde8;
        --qf-muted: #64748b;
        --qf-bg: #f8fafc;
      }
      .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
      }
      .qf-title {
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 0.15rem;
      }
      .qf-subtitle {
        color: var(--qf-muted);
        font-size: 0.92rem;
        margin-bottom: 1rem;
      }
      .qf-panel {
        border: 1px solid var(--qf-border);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        background: white;
      }
      .qf-section {
        font-weight: 650;
        font-size: 1rem;
        margin: 0.4rem 0 0.55rem 0;
      }
      .qf-muted {
        color: var(--qf-muted);
        font-size: 0.86rem;
      }
      iframe {
        border-radius: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def existing_report_dirs() -> list[Path]:
    if not REPORTS_DIR.exists():
        return []
    candidates = [
        path
        for path in REPORTS_DIR.iterdir()
        if path.is_dir() and (path / "performance_summary.csv").exists()
    ]
    return sorted(candidates, key=lambda path: path.name)


def existing_agent_runs() -> list[Path]:
    roots = [REPORTS_DIR / "agent_tool_runs", REPORTS_DIR / "agent_experiments"]
    runs: list[Path] = []
    for root in roots:
        if root.exists():
            runs.extend(path for path in root.iterdir() if path.is_dir())
    return sorted(runs, key=lambda path: path.name, reverse=True)


def existing_theme_reports() -> list[Path]:
    root = REPORTS_DIR / "themes"
    if not root.exists():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and (path / "trend_snapshot.csv").exists()],
        key=lambda path: path.name,
    )


def find_default_report(report_dirs: list[Path]) -> Path | None:
    if DEFAULT_REPORT in report_dirs:
        return DEFAULT_REPORT
    return report_dirs[0] if report_dirs else None


def dataframe_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_csv(str(path))


def text_or_empty(path: Path) -> str:
    if not path.exists():
        return ""
    return read_text(str(path))


def format_pct(value: float | int | str | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def format_float(value: float | int | str | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def strategy_row(performance: pd.DataFrame) -> pd.Series | None:
    if performance.empty or "symbol" not in performance.columns:
        return None
    preferred = performance.loc[performance["symbol"] == STRATEGY_SYMBOL]
    if not preferred.empty:
        return preferred.iloc[0]
    strategy_like = performance.loc[performance["symbol"].astype(str).str.contains("NET", na=False)]
    if not strategy_like.empty:
        return strategy_like.iloc[0]
    return performance.iloc[0]


def styled_metrics(row: pd.Series | None) -> None:
    if row is None:
        st.info("No performance metrics found for this report.")
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Annual Return", format_pct(row.get("annual_return")))
    col2.metric("Volatility", format_pct(row.get("annual_volatility")))
    col3.metric("Sharpe", format_float(row.get("sharpe")))
    col4.metric("Max Drawdown", format_pct(row.get("max_drawdown")))
    col5.metric("Calmar", format_float(row.get("calmar")))


def market_metrics(row: pd.Series | None) -> None:
    if row is None:
        st.info("No market state report found. Run `python run_market_regime.py` first.")
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Regime", str(row.get("regime", "n/a")))
    col2.metric("Posture", str(row.get("posture", "n/a")))
    col3.metric("Risk Score", format_float(row.get("risk_score"), digits=0))
    col4.metric("SPY 3M", format_pct(row.get("spy_3m_return")))
    col5.metric("SPY Vol", format_pct(row.get("spy_1m_volatility")))


def render_html_file(path: Path, height: int = 620) -> None:
    if not path.exists():
        st.info(f"Missing chart: {path.name}")
        return
    st.iframe(path, width="stretch", height=height)


def chart_file(report_dir: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = report_dir / name
        if candidate.exists():
            return candidate
    return None


def format_config(config: pd.DataFrame) -> pd.DataFrame:
    if config.empty:
        return config
    output = config.copy()
    output.columns = [str(column) for column in output.columns]
    return output


def show_table(dataframe: pd.DataFrame) -> None:
    output = dataframe.copy()
    for column in [name for name in output.columns if output[name].dtype == object]:
        output[column] = output[column].map(lambda value: "" if pd.isna(value) else str(value))
    st.dataframe(output, width="stretch", hide_index=True)


def show_overview(report_dir: Path) -> None:
    performance = dataframe_or_empty(report_dir / "performance_summary.csv")
    risk = dataframe_or_empty(report_dir / "risk_summary.csv")
    turnover = dataframe_or_empty(report_dir / "turnover_summary.csv")
    config = dataframe_or_empty(report_dir / "walkforward_config.csv")

    st.markdown('<div class="qf-section">Strategy Snapshot</div>', unsafe_allow_html=True)
    styled_metrics(strategy_row(performance))

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="qf-section">Performance</div>', unsafe_allow_html=True)
        if performance.empty:
            st.info("No performance table available.")
        else:
            show_table(performance)

    with right:
        st.markdown('<div class="qf-section">Risk And Turnover</div>', unsafe_allow_html=True)
        if not risk.empty:
            show_table(risk)
        if not turnover.empty:
            show_table(turnover)
        if not config.empty:
            with st.expander("Config", expanded=False):
                show_table(format_config(config))


def show_market_regime() -> None:
    report_dir = MARKET_REGIME_REPORT
    state = dataframe_or_empty(report_dir / "market_state.csv")
    relative_strength = dataframe_or_empty(report_dir / "relative_strength.csv")
    notes = dataframe_or_empty(report_dir / "market_notes.csv")
    config = dataframe_or_empty(report_dir / "market_regime_config.csv")
    summary = text_or_empty(report_dir / "market_regime_summary.md")

    if state.empty:
        st.info("No market regime report found. Run `conda run -n quantforge python run_market_regime.py`.")
        return

    st.markdown('<div class="qf-section">Current Market State</div>', unsafe_allow_html=True)
    market_metrics(state.iloc[0])

    if summary:
        with st.expander("Research Summary", expanded=False):
            st.markdown(summary)

    left, right = st.columns([2, 1])
    with left:
        st.markdown('<div class="qf-section">Relative Strength</div>', unsafe_allow_html=True)
        if relative_strength.empty:
            st.info("No relative strength table available.")
        else:
            show_table(relative_strength)
    with right:
        st.markdown('<div class="qf-section">Diagnostics</div>', unsafe_allow_html=True)
        if notes.empty:
            st.info("No diagnostic notes available.")
        else:
            for note in notes["note"]:
                st.markdown(f"- {note}")
        if not config.empty:
            with st.expander("Config", expanded=False):
                show_table(config)

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.markdown('<div class="qf-section">Regime Timeline</div>', unsafe_allow_html=True)
        render_html_file(report_dir / "regime_timeline.html", height=520)
    with chart_columns[1]:
        st.markdown('<div class="qf-section">Asset Trends</div>', unsafe_allow_html=True)
        render_html_file(report_dir / "asset_trends.html", height=520)

    st.markdown('<div class="qf-section">Market Indicators</div>', unsafe_allow_html=True)
    render_html_file(report_dir / "market_indicators.html", height=580)

    st.markdown('<div class="qf-section">Cross-Asset Correlation</div>', unsafe_allow_html=True)
    render_html_file(report_dir / "correlation_heatmap.html", height=680)


def ai_chip_metrics(snapshot: pd.DataFrame) -> None:
    if snapshot.empty:
        st.info("No AI chip trend snapshot found.")
        return

    leader = snapshot.iloc[0]
    benchmark = snapshot.loc[snapshot["symbol"] == "SMH"]
    benchmark_row = benchmark.iloc[0] if not benchmark.empty else None
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Leader", str(leader.get("symbol", "n/a")))
    col2.metric("Leader 2Y", format_pct(leader.get("total_return_2y")))
    col3.metric("Leader 3M", format_pct(leader.get("return_3m")))
    col4.metric("SMH 2Y", format_pct(benchmark_row.get("total_return_2y")) if benchmark_row is not None else "n/a")
    col5.metric("SMH 3M", format_pct(benchmark_row.get("return_3m")) if benchmark_row is not None else "n/a")


def show_ai_chip_trends() -> None:
    report_dir = AI_CHIP_REPORT
    snapshot = dataframe_or_empty(report_dir / "trend_snapshot.csv")
    group_summary = dataframe_or_empty(report_dir / "group_summary.csv")
    config = dataframe_or_empty(report_dir / "ai_chip_trend_config.csv")
    summary = text_or_empty(report_dir / "ai_chip_summary.md")

    if snapshot.empty:
        st.info(
            "No AI chip trend report found. Run `conda run -n quantforge python run_ai_chip_trends.py` "
            "after downloading `data/prices/ai_chip_daily.parquet`."
        )
        return

    st.markdown('<div class="qf-section">AI Chip Theme Snapshot</div>', unsafe_allow_html=True)
    ai_chip_metrics(snapshot)

    if summary:
        with st.expander("Research Summary", expanded=False):
            st.markdown(summary)

    left, right = st.columns([2, 1])
    with left:
        st.markdown('<div class="qf-section">Trend Snapshot</div>', unsafe_allow_html=True)
        show_table(snapshot)
    with right:
        st.markdown('<div class="qf-section">Industry Chain Groups</div>', unsafe_allow_html=True)
        if group_summary.empty:
            st.info("No group summary available.")
        else:
            show_table(group_summary)
        if not config.empty:
            with st.expander("Config", expanded=False):
                show_table(config)

    st.markdown('<div class="qf-section">Normalized Performance</div>', unsafe_allow_html=True)
    render_html_file(report_dir / "normalized_performance.html", height=650)

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.markdown('<div class="qf-section">Return Horizons</div>', unsafe_allow_html=True)
        render_html_file(report_dir / "return_horizon_bars.html", height=540)
    with chart_columns[1]:
        st.markdown('<div class="qf-section">Strength Heatmap</div>', unsafe_allow_html=True)
        render_html_file(report_dir / "strength_heatmap.html", height=540)

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.markdown('<div class="qf-section">Drawdowns</div>', unsafe_allow_html=True)
        render_html_file(report_dir / "drawdowns.html", height=560)
    with chart_columns[1]:
        st.markdown('<div class="qf-section">Correlation</div>', unsafe_allow_html=True)
        render_html_file(report_dir / "correlation_heatmap.html", height=560)


def theme_metrics(snapshot: pd.DataFrame) -> None:
    if snapshot.empty:
        st.info("No theme trend snapshot found.")
        return
    leader = snapshot.iloc[0]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Leader", str(leader.get("symbol", "n/a")))
    col2.metric("Lookback", format_pct(leader.get("total_return")))
    col3.metric("3M", format_pct(leader.get("return_3m")))
    col4.metric("Vol", format_pct(leader.get("volatility_recent")))
    col5.metric("Drawdown", format_pct(leader.get("max_drawdown")))


def show_theme_reports() -> None:
    reports = existing_theme_reports()
    if not reports:
        st.info("No generic theme reports found. Run `conda run -n quantforge python run_theme_trends.py --theme data/themes/ai_chips.yaml`.")
        return

    selected = st.selectbox("Theme Report", reports, format_func=lambda path: path.name)
    snapshot = dataframe_or_empty(selected / "trend_snapshot.csv")
    group_summary = dataframe_or_empty(selected / "group_summary.csv")
    config = dataframe_or_empty(selected / "theme_trend_config.csv")
    summary = text_or_empty(selected / "theme_summary.md")

    st.markdown('<div class="qf-section">Theme Snapshot</div>', unsafe_allow_html=True)
    theme_metrics(snapshot)

    if summary:
        with st.expander("Research Summary", expanded=False):
            st.markdown(summary)

    left, right = st.columns([2, 1])
    with left:
        st.markdown('<div class="qf-section">Trend Snapshot</div>', unsafe_allow_html=True)
        if snapshot.empty:
            st.info("No trend snapshot available.")
        else:
            show_table(snapshot)
    with right:
        st.markdown('<div class="qf-section">Groups</div>', unsafe_allow_html=True)
        if group_summary.empty:
            st.info("No group summary available.")
        else:
            show_table(group_summary)
        if not config.empty:
            with st.expander("Config", expanded=False):
                show_table(config)

    st.markdown('<div class="qf-section">Normalized Performance</div>', unsafe_allow_html=True)
    render_html_file(selected / "normalized_performance.html", height=650)

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.markdown('<div class="qf-section">Return Horizons</div>', unsafe_allow_html=True)
        render_html_file(selected / "return_horizon_bars.html", height=540)
    with chart_columns[1]:
        st.markdown('<div class="qf-section">Strength Heatmap</div>', unsafe_allow_html=True)
        render_html_file(selected / "strength_heatmap.html", height=540)

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.markdown('<div class="qf-section">Drawdowns</div>', unsafe_allow_html=True)
        render_html_file(selected / "drawdowns.html", height=560)
    with chart_columns[1]:
        st.markdown('<div class="qf-section">Correlation</div>', unsafe_allow_html=True)
        render_html_file(selected / "correlation_heatmap.html", height=560)


def show_strategy(report_dir: Path) -> None:
    chart = chart_file(report_dir, ["equity_curve.html", "equity_curves.html", "multifactor_equity_curve.html"])
    if chart:
        render_html_file(chart, height=650)

    col1, col2 = st.columns([2, 1])
    with col1:
        holdings = dataframe_or_empty(report_dir / "holdings.csv")
        st.markdown('<div class="qf-section">Recent Holdings</div>', unsafe_allow_html=True)
        if holdings.empty:
            st.info("No holdings table available.")
        else:
            show_table(holdings.tail(36))
    with col2:
        cumulative = dataframe_or_empty(report_dir / "cumulative_wealth.csv")
        st.markdown('<div class="qf-section">Final Wealth</div>', unsafe_allow_html=True)
        if cumulative.empty:
            st.info("No cumulative wealth table available.")
        else:
            last = cumulative.tail(1).T.reset_index()
            last.columns = ["series", "final_value"]
            show_table(last)

    heatmap = chart_file(report_dir, ["weights_heatmap.html"])
    if heatmap:
        st.markdown('<div class="qf-section">Weights</div>', unsafe_allow_html=True)
        render_html_file(heatmap, height=580)


def show_factors(report_dir: Path) -> None:
    summary = dataframe_or_empty(report_dir / "factor_selection_summary.csv")
    if not summary.empty:
        st.markdown('<div class="qf-section">Factor Selection Summary</div>', unsafe_allow_html=True)
        show_table(summary)

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.markdown('<div class="qf-section">Factor Selection</div>', unsafe_allow_html=True)
        render_html_file(report_dir / "factor_selection_heatmap.html", height=520)
    with chart_columns[1]:
        st.markdown('<div class="qf-section">Cluster Representatives</div>', unsafe_allow_html=True)
        render_html_file(report_dir / "cluster_representative_heatmap.html", height=520)

    rolling_ic = chart_file(report_dir, ["rolling_ic.html", "ic_heatmap.html"])
    if rolling_ic:
        st.markdown('<div class="qf-section">Rolling IC</div>', unsafe_allow_html=True)
        render_html_file(rolling_ic, height=620)

    factor_values = dataframe_or_empty(report_dir / "factor_values_with_forward_returns.csv")
    if not factor_values.empty:
        with st.expander("Factor Values Sample", expanded=False):
            show_table(factor_values.tail(200))


def show_experiments() -> None:
    runs = [path for path in existing_agent_runs() if (path / "summary.md").exists() or (path / "agent_recap.md").exists()]
    if not runs:
        st.info("No agent tool runs found.")
        return

    selected = st.selectbox("Agent Run", runs, format_func=lambda path: path.name)
    summary = text_or_empty(selected / "summary.md")
    recap = text_or_empty(selected / "agent_recap.md")
    results = dataframe_or_empty(selected / "experiment_results.csv")
    plan_text = text_or_empty(selected / "validated_plan.json")

    if recap:
        st.markdown('<div class="qf-section">Agent Recap</div>', unsafe_allow_html=True)
        st.markdown(recap)

    if summary:
        st.markdown('<div class="qf-section">Run Summary</div>', unsafe_allow_html=True)
        st.markdown(summary)

    if not results.empty:
        st.markdown('<div class="qf-section">Experiment Results</div>', unsafe_allow_html=True)
        show_table(results)

    if plan_text:
        with st.expander("Validated Plan", expanded=False):
            try:
                st.json(json.loads(plan_text))
            except json.JSONDecodeError:
                st.code(plan_text)


def show_agent_reports() -> None:
    runs = [path for path in existing_agent_runs() if (path / "agent_report.md").exists()]
    if not runs:
        st.info("No standalone agent reports found.")
        return

    selected = st.selectbox("Agent Report", runs, format_func=lambda path: path.name)
    st.markdown(text_or_empty(selected / "agent_report.md"))
    metadata = text_or_empty(selected / "metadata.json")
    if metadata:
        with st.expander("Metadata", expanded=False):
            st.json(json.loads(metadata))


def main() -> None:
    st.markdown('<div class="qf-title">QuantForge Research Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="qf-subtitle">ETF strategy research, factor diagnostics, and AI agent experiment logs.</div>',
        unsafe_allow_html=True,
    )

    report_dirs = existing_report_dirs()
    default_report = find_default_report(report_dirs)
    if default_report is None:
        st.error("No report directories found under reports/.")
        return

    with st.sidebar:
        st.markdown("### Report")
        report_dir = st.selectbox(
            "Strategy Report",
            report_dirs,
            index=report_dirs.index(default_report),
            format_func=lambda path: path.name,
        )
        st.caption(str(report_dir))

    market_tab, chip_tab, themes_tab, overview_tab, strategy_tab, factor_tab, experiments_tab, agent_tab = st.tabs(
        ["Market", "AI Chips", "Themes", "Overview", "Strategy", "Factors", "Experiments", "Agent Reports"]
    )
    with market_tab:
        show_market_regime()
    with chip_tab:
        show_ai_chip_trends()
    with themes_tab:
        show_theme_reports()
    with overview_tab:
        show_overview(report_dir)
    with strategy_tab:
        show_strategy(report_dir)
    with factor_tab:
        show_factors(report_dir)
    with experiments_tab:
        show_experiments()
    with agent_tab:
        show_agent_reports()


if __name__ == "__main__":
    main()
