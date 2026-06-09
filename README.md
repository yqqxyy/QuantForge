# QuantForge

QuantForge is a research-oriented quantitative investing workbench. It combines ETF allocation, factor research, walk-forward validation, market regime analysis, theme monitoring, an optional local LLM research agent, and a Streamlit dashboard.

QuantForge 是一个量化研究工作台，集成了 ETF 配置、因子研究、walk-forward 样本外验证、市场状态分析、主题趋势观察、可选本地 LLM 研究助理，以及 Streamlit 可视化面板。

> Disclaimer: this project is for research and education only. It is not investment advice.
>
> 免责声明：本项目仅用于研究和学习，不构成投资建议。

## Highlights

- Download daily OHLCV data from Yahoo Finance with `yfinance`.
- Run ETF buy-and-hold benchmarks, momentum rotation, factor research, and walk-forward factor strategies.
- Add risk overlays such as market filters, volatility targeting, transaction costs, and drawdown diagnostics.
- Analyze market regimes across equities, bonds, commodities, and sectors.
- Monitor arbitrary themes such as AI chips, oil/geopolitics, uranium/nuclear, solar, batteries, or cybersecurity.
- Automatically discover candidate tickers for a theme and generate theme YAML configs.
- Use an optional local LLM agent to propose experiments and summarize research results.
- Explore results in a Streamlit dashboard.

## Repository Layout

```text
.
├── data/
│   ├── universe_etf.yaml              # Default ETF universe
│   ├── universe_ai_chips.yaml         # AI chip theme universe
│   ├── themes/                        # Generic theme configs
│   └── prices/                        # Local parquet price files, ignored by git
├── reports/                           # Generated reports, ignored by git
├── download_prices.py                 # yfinance downloader
├── analyze_prices.py                  # Basic buy-and-hold analysis
├── run_momentum_strategy.py           # Monthly ETF momentum rotation
├── scan_momentum_params.py            # Momentum parameter scan
├── run_factor_research.py             # Factor IC and multi-factor research
├── run_walkforward_factor_strategy.py # Walk-forward IC-weighted factor strategy
├── run_market_regime.py               # Market regime diagnostics
├── run_theme_trends.py                # Generic theme trend monitor
├── discover_theme_tickers.py          # Theme ticker discovery
├── create_theme_config.py             # Theme YAML generator from known tickers
├── research_agent.py                  # Optional local LLM research report agent
├── agent_tool_runner.py               # Optional local LLM experiment runner
├── dashboard.py                       # Streamlit dashboard
└── local_llm.py                       # Local LLM wrapper
```

## Installation

Create a Python environment:

```bash
conda create -n quantforge python=3.11 -y
conda activate quantforge
pip install pandas numpy yfinance duckdb pyarrow pyyaml plotly streamlit
```

Optional local LLM support currently expects `mlx-lm` and a compatible local model. Install it in the same environment or in a separate environment:

```bash
pip install mlx-lm
```

You can configure the default local model and cache path with environment variables:

```bash
export QUANTFORGE_LLM_MODEL="mlx-community/Qwen3.5-4B-OptiQ-4bit"
export QUANTFORGE_HF_HOME="$HOME/.cache/huggingface"
```

If you do not need the LLM agent, you can skip the optional LLM setup.

## Quick Start

Download the default ETF universe:

```bash
conda run -n quantforge python download_prices.py
```

Run market regime analysis:

```bash
conda run -n quantforge python run_market_regime.py
```

Run the main walk-forward factor strategy:

```bash
conda run -n quantforge python run_walkforward_factor_strategy.py \
  --decorrelate-factors \
  --report-dir reports/walkforward_factor_clustered_loose
```

Start the dashboard:

```bash
conda run -n quantforge streamlit run dashboard.py \
  --server.port 8501 \
  --server.address 127.0.0.1 \
  --server.headless true \
  --browser.gatherUsageStats false
```

Open:

```text
http://127.0.0.1:8501
```

Stop a running dashboard:

```bash
lsof -i :8501
kill <PID>
```

## Data

The default downloader reads a YAML universe and writes a parquet file.

Default ETF data:

```bash
conda run -n quantforge python download_prices.py
```

Specific symbols:

```bash
conda run -n quantforge python download_prices.py --symbols SPY QQQ GLD
```

Theme-specific data:

```bash
conda run -n quantforge python download_prices.py \
  --universe data/themes/oil_geopolitics.yaml \
  --start 2024-01-01 \
  --output data/prices/oil_geopolitics_daily.parquet
```

Check the latest date:

```bash
conda run -n quantforge python -c "import pandas as pd; df=pd.read_parquet('data/prices/etf_daily.parquet'); print(df['date'].max()); print(df['symbol'].nunique(), len(df))"
```

Notes:

- Yahoo Finance data is accessed through `yfinance`.
- Stocks, ETFs, and many futures symbols are supported.
- Non-market data such as vegetable prices, rents, electricity prices, or shipping rates should first be imported into the same parquet schema.

## Strategies And Research Modules

### Buy-And-Hold Baselines

```bash
conda run -n quantforge python analyze_prices.py
```

Outputs include data quality checks, daily returns, cumulative wealth, and performance metrics.

### Momentum Rotation

Monthly ETF rotation based on trailing momentum:

```bash
conda run -n quantforge python run_momentum_strategy.py
```

Example with a market filter and volatility target:

```bash
conda run -n quantforge python run_momentum_strategy.py \
  --lookback-days 252 \
  --top-k 3 \
  --market-filter-symbol SPY \
  --target-volatility 0.12 \
  --report-dir reports/momentum_rotation_risk_control
```

### Parameter Scan

```bash
conda run -n quantforge python scan_momentum_params.py
```

This scans momentum lookbacks, portfolio sizes, market filters, and volatility targets.

### Factor Research

```bash
conda run -n quantforge python run_factor_research.py
```

Current factors include:

- `momentum_63`
- `momentum_126`
- `momentum_252`
- `trend_200`
- `volatility_63`
- `volatility_126`
- `downside_volatility_63`
- `max_drawdown_126`

The module computes IC, rank IC, quantile returns, factor correlations, fixed-prior multi-factor scores, and IC-weighted scores.

### Walk-Forward Factor Strategy

```bash
conda run -n quantforge python run_walkforward_factor_strategy.py \
  --decorrelate-factors \
  --report-dir reports/walkforward_factor_clustered_loose
```

Features:

- Rolling IC estimation
- Factor stability filtering
- Factor decorrelation / clustering
- Cluster representative selection
- Market filter support
- Volatility targeting
- Transaction cost modeling

Useful options:

```bash
--forward-days 63
--train-months 36
--min-train-months 18
--min-ic-months 12
--min-abs-ic 0.015
--min-directional-ic-frac 0.55
--min-abs-ic-ir 0.05
--decorrelate-factors
--factor-corr-threshold 0.8
--top-k 3
--market-filter-symbol SPY
--target-volatility 0.12
```

## Market Regime Analysis

```bash
conda run -n quantforge python run_market_regime.py
```

The module classifies the current market as:

- `risk_on`
- `mixed`
- `risk_off`

It evaluates equity momentum, SPY versus its 200-day moving average, equities versus Treasuries, growth versus broad equity, small caps versus large caps, gold versus equities, cyclical sectors versus defensive sectors, volatility, drawdown, and relative strength.

Results are written to:

```text
reports/market_regime/
```

The dashboard reads these outputs in the `Market` tab.

## Theme Monitor

The generic theme monitor analyzes a configurable basket of stocks, ETFs, or futures.

Built-in example configs:

```text
data/themes/ai_chips.yaml
data/themes/oil_geopolitics.yaml
data/themes/vegetables_template.yaml
```

Run AI chip analysis:

```bash
conda run -n quantforge python download_prices.py \
  --universe data/themes/ai_chips.yaml \
  --start 2024-01-01 \
  --output data/prices/ai_chip_daily.parquet

conda run -n quantforge python run_theme_trends.py --theme data/themes/ai_chips.yaml
```

Run oil/geopolitics analysis:

```bash
conda run -n quantforge python download_prices.py \
  --universe data/themes/oil_geopolitics.yaml \
  --start 2024-01-01 \
  --output data/prices/oil_geopolitics_daily.parquet

conda run -n quantforge python run_theme_trends.py --theme data/themes/oil_geopolitics.yaml
```

Theme reports are written to:

```text
reports/themes/<slug>/
```

The dashboard reads them in the `Themes` tab.

## Theme Config Generation

If you already know the ticker list:

```bash
conda run -n quantforge python create_theme_config.py \
  --name "Uranium And Nuclear Theme" \
  --symbols URA URNM CCJ UEC NXE DNN SMR CEG \
  --benchmark URA \
  --leader CCJ \
  --description "Uranium miners, nuclear energy ETFs, and nuclear infrastructure stocks."
```

If you only know the theme name, let the discovery tool propose candidates:

```bash
conda run -n quantforge python discover_theme_tickers.py --theme-name "AI chip"
conda run -n quantforge python discover_theme_tickers.py --theme-name "uranium nuclear"
conda run -n quantforge python discover_theme_tickers.py --theme-name "oil geopolitics"
```

Outputs:

```text
data/themes/<slug>.yaml
reports/theme_discovery/<slug>/candidates.csv
reports/theme_discovery/<slug>/selected.csv
```

Then run the printed download and analysis commands.

Useful options:

- `--seed-symbols`: force include specific tickers.
- `--queries`: add extra search queries.
- `--max-symbols`: control basket size.
- `--benchmark`: override the theme benchmark.
- `--leader`: override the theme leader.

Ticker discovery is heuristic. Always review the generated universe before using it for research.

## Optional AI Agent

QuantForge includes two optional local LLM entry points:

```bash
conda run -n quantforge python research_agent.py --max-tokens 1800
conda run -n quantforge python agent_tool_runner.py
```

If your LLM dependencies live in a separate environment, run those scripts from that environment instead.

Common modes:

```bash
conda run -n quantforge python agent_tool_runner.py --plan-only
conda run -n quantforge python agent_tool_runner.py --use-fallback-plan --max-experiments 1
conda run -n quantforge python agent_tool_runner.py --skip-recap
```

The agent can:

- Read baseline strategy reports.
- Propose new experiment parameters.
- Validate parameters against a whitelist.
- Run walk-forward strategy experiments.
- Compare results with the baseline.
- Generate experiment summaries.

Outputs:

```text
reports/agent_experiments/
reports/agent_tool_runs/
```

## Dashboard

Start:

```bash
conda run -n quantforge streamlit run dashboard.py \
  --server.port 8501 \
  --server.address 127.0.0.1 \
  --server.headless true \
  --browser.gatherUsageStats false
```

Open:

```text
http://127.0.0.1:8501
```

Tabs:

- `Market`: market regime diagnostics
- `AI Chips`: AI chip theme monitor
- `Themes`: generic theme reports
- `Overview`: selected strategy performance
- `Strategy`: equity curves, holdings, and weights
- `Factors`: factor selection, clusters, and rolling IC
- `Experiments`: AI experiment logs
- `Agent Reports`: standalone agent reports

## Public Sharing With Ngrok

Start the dashboard first, then expose it:

```bash
ngrok http http://127.0.0.1:8501
```

Inspect the generated public URL:

```bash
curl http://127.0.0.1:4040/api/tunnels
```

Stop ngrok:

```bash
lsof -i :4040
kill <PID>
```

Security notes:

- Free ngrok URLs are usually temporary.
- The dashboard has no built-in authentication.
- Do not expose secrets, account data, tokens, or private holdings.

## Common Workflows

ETF strategy research:

```bash
conda run -n quantforge python download_prices.py
conda run -n quantforge python run_market_regime.py
conda run -n quantforge python run_walkforward_factor_strategy.py --decorrelate-factors --report-dir reports/walkforward_factor_clustered_loose
conda run -n quantforge streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

Theme research:

```bash
conda run -n quantforge python discover_theme_tickers.py --theme-name "uranium nuclear"
conda run -n quantforge python download_prices.py --universe data/themes/uranium_nuclear.yaml --start 2024-01-01 --output data/prices/uranium_nuclear_daily.parquet
conda run -n quantforge python run_theme_trends.py --theme data/themes/uranium_nuclear.yaml
conda run -n quantforge streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

AI-assisted experiment loop:

```bash
conda run -n quantforge python run_walkforward_factor_strategy.py --decorrelate-factors --report-dir reports/walkforward_factor_clustered_loose
conda run -n quantforge python agent_tool_runner.py
conda run -n quantforge streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

## Generated Report Files

Common outputs:

- `performance_summary.csv`
- `risk_summary.csv`
- `turnover_summary.csv`
- `holdings.csv`
- `daily_weights.csv`
- `cumulative_wealth.csv`
- `*.html` Plotly charts
- `*_summary.md` readable summaries

Generated reports and local price data are ignored by git.

## Roadmap Ideas

- ML-based cross-sectional ranking strategies
- Fundamental data integration
- News, filings, and macro event features
- Non-market data adapters, such as USDA market data
- LLM-assisted theme discovery with citations
- More robust portfolio risk models
- Transaction cost, slippage, and liquidity constraints
- Authenticated deployment for sharing dashboards

---

# 中文说明

QuantForge 是一个面向学习、研究和展示的量化投资工作台。它包含 ETF 轮动、因子研究、walk-forward 样本外验证、市场状态分析、主题趋势观察、可选本地 LLM agent 和 Streamlit dashboard。

本项目仅用于研究和学习，不构成投资建议。

## 功能概览

- 使用 `yfinance` 下载股票、ETF、期货日线数据。
- 支持 buy-and-hold 基准、动量轮动、多因子研究、walk-forward 因子策略。
- 支持市场过滤、波动率目标、交易成本和回撤分析。
- 支持跨资产市场状态判断。
- 支持 AI 芯片、原油/地缘政治、核电/铀矿等主题趋势观察。
- 可以根据主题名自动发现候选 ticker 并生成主题配置。
- 可选使用本地 LLM 生成研究报告和自动实验计划。
- 使用 Streamlit dashboard 查看结果。

## 安装

```bash
conda create -n quantforge python=3.11 -y
conda activate quantforge
pip install pandas numpy yfinance duckdb pyarrow pyyaml plotly streamlit
```

如果需要本地 LLM agent，可以额外安装：

```bash
pip install mlx-lm
```

本地 LLM 可通过环境变量配置：

```bash
export QUANTFORGE_LLM_MODEL="mlx-community/Qwen3.5-4B-OptiQ-4bit"
export QUANTFORGE_HF_HOME="$HOME/.cache/huggingface"
```

不使用 AI agent 时可以跳过这一步。

## 快速开始

下载默认 ETF 数据：

```bash
conda run -n quantforge python download_prices.py
```

运行市场状态分析：

```bash
conda run -n quantforge python run_market_regime.py
```

运行主策略：

```bash
conda run -n quantforge python run_walkforward_factor_strategy.py \
  --decorrelate-factors \
  --report-dir reports/walkforward_factor_clustered_loose
```

启动 dashboard：

```bash
conda run -n quantforge streamlit run dashboard.py \
  --server.port 8501 \
  --server.address 127.0.0.1 \
  --server.headless true \
  --browser.gatherUsageStats false
```

打开：

```text
http://127.0.0.1:8501
```

关闭：

```bash
lsof -i :8501
kill <PID>
```

## 常用模块

基础分析：

```bash
conda run -n quantforge python analyze_prices.py
```

动量轮动：

```bash
conda run -n quantforge python run_momentum_strategy.py
```

参数扫描：

```bash
conda run -n quantforge python scan_momentum_params.py
```

因子研究：

```bash
conda run -n quantforge python run_factor_research.py
```

Walk-forward 因子策略：

```bash
conda run -n quantforge python run_walkforward_factor_strategy.py --decorrelate-factors --report-dir reports/walkforward_factor_clustered_loose
```

市场状态分析：

```bash
conda run -n quantforge python run_market_regime.py
```

## 主题观察台

如果已经有主题配置：

```bash
conda run -n quantforge python download_prices.py \
  --universe data/themes/ai_chips.yaml \
  --start 2024-01-01 \
  --output data/prices/ai_chip_daily.parquet

conda run -n quantforge python run_theme_trends.py --theme data/themes/ai_chips.yaml
```

如果只知道主题名，可以自动发现 ticker：

```bash
conda run -n quantforge python discover_theme_tickers.py --theme-name "AI chip"
conda run -n quantforge python discover_theme_tickers.py --theme-name "uranium nuclear"
conda run -n quantforge python discover_theme_tickers.py --theme-name "oil geopolitics"
```

自动发现会生成：

```text
data/themes/<slug>.yaml
reports/theme_discovery/<slug>/candidates.csv
reports/theme_discovery/<slug>/selected.csv
```

自动发现是启发式工具，生成的 ticker 列表需要人工检查。

## 可选 AI Agent

研究报告 agent：

```bash
conda run -n quantforge python research_agent.py --max-tokens 1800
```

自动实验 agent：

```bash
conda run -n quantforge python agent_tool_runner.py
```

如果 LLM 依赖安装在单独环境中，可以用对应环境运行这两个脚本。

## Dashboard 页面

- `Market`：市场状态分析
- `AI Chips`：AI 芯片主题
- `Themes`：通用主题观察台
- `Overview`：策略指标总览
- `Strategy`：净值、持仓和权重
- `Factors`：因子筛选、聚类和 rolling IC
- `Experiments`：AI agent 实验记录
- `Agent Reports`：AI 研究报告

## Ngrok 公网分享

先启动 dashboard，然后：

```bash
ngrok http http://127.0.0.1:8501
```

查看公网 URL：

```bash
curl http://127.0.0.1:4040/api/tunnels
```

关闭 ngrok：

```bash
lsof -i :4040
kill <PID>
```

注意：dashboard 没有内置登录保护，不要公开敏感数据。

## 输出文件

常见报告：

- `performance_summary.csv`
- `risk_summary.csv`
- `turnover_summary.csv`
- `holdings.csv`
- `daily_weights.csv`
- `cumulative_wealth.csv`
- `*.html` 图表
- `*_summary.md` 文字总结

`reports/` 和本地行情数据默认被 `.gitignore` 忽略。
