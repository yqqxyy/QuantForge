# QuantForge

QuantForge 是一个面向学习和展示的量化研究项目。它把 ETF 轮动、因子研究、walk-forward 样本外验证、市场状态判断、主题趋势观察、AI agent 实验生成和 Streamlit dashboard 放在同一个工作台里。

这个项目适合用来学习：

- 金融市场数据的获取和清洗
- ETF / 股票 / 期货主题篮子的趋势分析
- 动量策略、多因子策略、风险控制和参数扫描
- walk-forward 验证、因子稳定性筛选、因子去相关 / 聚类
- 用本地大模型做量化研究助理
- 用 dashboard 和 ngrok 展示研究结果

注意：本项目是研究和学习工具，不构成投资建议。

## 项目结构

```text
.
├── data/
│   ├── universe_etf.yaml              # 默认 ETF 股票池
│   ├── universe_ai_chips.yaml         # AI 芯片主题股票池
│   ├── themes/                        # 通用主题配置
│   └── prices/                        # 下载后的 parquet 行情数据
├── reports/                           # 所有策略和分析报告
├── download_prices.py                 # 下载 yfinance 日线行情
├── analyze_prices.py                  # 基础收益统计和 buy-and-hold 基准
├── run_momentum_strategy.py           # ETF 月度动量轮动策略
├── scan_momentum_params.py            # 动量策略参数扫描
├── run_factor_research.py             # 因子研究和多因子策略
├── run_walkforward_factor_strategy.py # walk-forward 因子策略
├── run_market_regime.py               # 市场状态分析
├── run_theme_trends.py                # 通用主题趋势观察台
├── discover_theme_tickers.py          # 根据主题名自动发现 ticker
├── create_theme_config.py             # 根据 ticker 列表生成主题配置
├── research_agent.py                  # 本地 Qwen 研究报告 agent
├── agent_tool_runner.py               # AI 自动提出、运行、比较实验
├── dashboard.py                       # Streamlit dashboard
└── local_llm.py                       # 本地 Qwen / MLX wrapper
```

## 环境

主要量化代码使用 `quantforge` 环境。当前项目依赖大致包括：

```bash
conda create -n quantforge python=3.11 -y
conda activate quantforge
pip install pandas numpy yfinance duckdb pyarrow pyyaml plotly streamlit
```

本地 Qwen agent 使用你已有的 `chunqiu` 环境，因为 Qwen/MLX 已经装在那里。

常用执行格式：

```bash
conda run -n quantforge python <script.py>
conda run -n chunqiu python <agent_script.py>
```

## 快速开始

先下载默认 ETF 数据：

```bash
conda run -n quantforge python download_prices.py
```

它会读取：

```text
data/universe_etf.yaml
```

并保存到：

```text
data/prices/etf_daily.parquet
```

然后跑市场状态分析和当前主策略：

```bash
conda run -n quantforge python run_market_regime.py
conda run -n quantforge python run_walkforward_factor_strategy.py --decorrelate-factors --report-dir reports/walkforward_factor_clustered_loose
```

启动 dashboard：

```bash
conda run -n quantforge streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

浏览器打开：

```text
http://127.0.0.1:8501
```

关闭 dashboard：

```bash
lsof -i :8501
kill <PID>
```

## 数据下载

下载默认 ETF 股票池：

```bash
conda run -n quantforge python download_prices.py
```

只下载指定 ticker：

```bash
conda run -n quantforge python download_prices.py --symbols SPY QQQ GLD
```

下载某个主题配置里的 ticker：

```bash
conda run -n quantforge python download_prices.py --universe data/themes/oil_geopolitics.yaml --start 2024-01-01 --output data/prices/oil_geopolitics_daily.parquet
```

检查数据最新日期：

```bash
conda run -n quantforge python -c "import pandas as pd; df=pd.read_parquet('data/prices/etf_daily.parquet'); print(df['date'].max()); print(df['symbol'].nunique(), len(df))"
```

说明：

- 默认数据源是 Yahoo Finance，通过 `yfinance` 获取。
- 股票、ETF、部分期货可以直接下载。
- 蔬菜价格、房租、电价等非金融 ticker 需要先接入外部数据源，再转换成统一 parquet 格式。

## 基础分析

基础 buy-and-hold 和等权组合分析：

```bash
conda run -n quantforge python analyze_prices.py
```

输出：

```text
reports/data_quality.csv
reports/performance_summary.csv
reports/daily_returns.csv
reports/cumulative_wealth.csv
reports/equity_curves.html
```

## 动量轮动策略

脚本：

```text
run_momentum_strategy.py
```

核心逻辑：

> 每个月看过去一段时间表现最好的 ETF，买入排名靠前的若干个。

默认运行：

```bash
conda run -n quantforge python run_momentum_strategy.py
```

示例：252 日动量、持有前 3 名、加入 SPY 市场过滤和 12% 波动率目标：

```bash
conda run -n quantforge python run_momentum_strategy.py \
  --lookback-days 252 \
  --top-k 3 \
  --market-filter-symbol SPY \
  --target-volatility 0.12 \
  --report-dir reports/momentum_rotation_risk_control
```

常用参数：

- `--lookback-days`：动量观察窗口
- `--top-k`：每次持有前几名
- `--market-filter-symbol SPY`：SPY 跌破均线时降低风险
- `--target-volatility`：目标波动率控制
- `--transaction-cost-bps`：交易成本

## 参数扫描

脚本：

```text
scan_momentum_params.py
```

运行：

```bash
conda run -n quantforge python scan_momentum_params.py
```

它会扫描：

- 不同动量窗口
- 不同持仓数量
- 是否加市场过滤
- 是否加波动率目标

输出：

```text
reports/parameter_scan/momentum_scan_results.csv
reports/parameter_scan/best_by_sharpe.csv
reports/parameter_scan/best_by_calmar.csv
reports/parameter_scan/risk_return_scatter.html
```

## 因子研究

脚本：

```text
run_factor_research.py
```

运行：

```bash
conda run -n quantforge python run_factor_research.py
```

当前因子包括：

- `momentum_63`
- `momentum_126`
- `momentum_252`
- `trend_200`
- `volatility_63`
- `volatility_126`
- `downside_volatility_63`
- `max_drawdown_126`

它会计算：

- IC / Rank IC
- 分位数组合收益
- 因子相关性
- 固定先验多因子策略
- IC-weighted 多因子策略

输出目录：

```text
reports/factor_research/
```

## Walk-Forward 因子策略

脚本：

```text
run_walkforward_factor_strategy.py
```

这是目前项目里最接近真实量化研究流程的策略。它模拟：

> 用过去训练/筛选因子，用未来样本外验证策略。

推荐运行：

```bash
conda run -n quantforge python run_walkforward_factor_strategy.py \
  --decorrelate-factors \
  --report-dir reports/walkforward_factor_clustered_loose
```

它支持：

- rolling IC 估计
- 因子稳定性筛选
- 因子去相关 / 因子聚类
- 每个相关因子簇保留代表因子
- SPY 市场过滤
- 目标波动率控制
- 交易成本

常用参数：

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

## 市场状态分析

脚本：

```text
run_market_regime.py
```

运行：

```bash
conda run -n quantforge python run_market_regime.py
```

它判断当前市场：

- `risk_on`
- `mixed`
- `risk_off`

并分析：

- SPY 是否在 200 日均线上方
- 股票 vs 长债
- 科技 vs 大盘
- 小盘 vs 大盘
- 黄金 vs 股票
- 周期板块 vs 防御板块
- 各 ETF 的相对强弱、波动率、回撤

输出：

```text
reports/market_regime/
```

dashboard 里的 `Market` tab 会读取这些结果。

## 主题观察台

通用脚本：

```text
run_theme_trends.py
```

它适合观察一个主题篮子，比如：

- AI 芯片
- 原油 / 地缘政治
- 核电 / 铀矿
- 网络安全
- 太阳能
- 电池 / 锂矿

已经有的主题配置：

```text
data/themes/ai_chips.yaml
data/themes/oil_geopolitics.yaml
data/themes/vegetables_template.yaml
```

运行 AI 芯片主题：

```bash
conda run -n quantforge python download_prices.py --universe data/themes/ai_chips.yaml --start 2024-01-01 --output data/prices/ai_chip_daily.parquet
conda run -n quantforge python run_theme_trends.py --theme data/themes/ai_chips.yaml
```

运行原油主题：

```bash
conda run -n quantforge python download_prices.py --universe data/themes/oil_geopolitics.yaml --start 2024-01-01 --output data/prices/oil_geopolitics_daily.parquet
conda run -n quantforge python run_theme_trends.py --theme data/themes/oil_geopolitics.yaml
```

输出：

```text
reports/themes/<slug>/
```

dashboard 里的 `Themes` tab 会自动列出这些主题报告。

## 自动生成主题配置

如果你已经知道 ticker 列表，用：

```bash
conda run -n quantforge python create_theme_config.py \
  --name "Uranium And Nuclear Theme" \
  --symbols URA URNM CCJ UEC NXE DNN SMR CEG \
  --benchmark URA \
  --leader CCJ \
  --description "Uranium miners, nuclear energy ETFs, and nuclear infrastructure stocks."
```

如果你只知道主题名，让系统自动找 ticker，用：

```bash
conda run -n quantforge python discover_theme_tickers.py --theme-name "AI chip"
```

或者：

```bash
conda run -n quantforge python discover_theme_tickers.py --theme-name "uranium nuclear"
conda run -n quantforge python discover_theme_tickers.py --theme-name "oil geopolitics"
```

它会生成：

```text
data/themes/<slug>.yaml
reports/theme_discovery/<slug>/candidates.csv
reports/theme_discovery/<slug>/selected.csv
```

然后继续跑它打印出的两条命令：

```bash
conda run -n quantforge python download_prices.py --universe data/themes/<slug>.yaml --start 2024-01-01 --output data/prices/<slug>_daily.parquet
conda run -n quantforge python run_theme_trends.py --theme data/themes/<slug>.yaml
```

说明：

- 自动发现依赖 Yahoo Finance 搜索和内置主题提示词。
- 结果应人工检查，尤其是小众主题。
- 可以用 `--seed-symbols` 强制加入你关心的 ticker。
- 可以用 `--queries` 加额外搜索词。
- 可以用 `--max-symbols` 控制候选数量。

## AI Agent

本地 LLM 配置在：

```text
local_llm.py
```

默认使用：

```text
mlx-community/Qwen3.5-4B-OptiQ-4bit
/Users/qqxyyy/DeepLearning/chunqiu/run/hf_cache
```

研究报告 agent：

```bash
conda run -n chunqiu python research_agent.py --max-tokens 1800
```

自动实验 agent：

```bash
conda run -n chunqiu python agent_tool_runner.py
```

常用模式：

```bash
conda run -n chunqiu python agent_tool_runner.py --plan-only
conda run -n chunqiu python agent_tool_runner.py --use-fallback-plan --max-experiments 1
conda run -n chunqiu python agent_tool_runner.py --skip-recap
```

AI agent 会：

- 读取 baseline 策略报告
- 提出实验参数
- 校验参数是否安全
- 调用 walk-forward 策略脚本
- 对比实验和 baseline
- 生成总结报告

输出：

```text
reports/agent_experiments/
reports/agent_tool_runs/
```

## Dashboard

启动：

```bash
conda run -n quantforge streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

访问：

```text
http://127.0.0.1:8501
```

当前页面：

- `Market`：市场状态分析
- `AI Chips`：AI 芯片主题观察
- `Themes`：通用主题观察台
- `Overview`：策略指标总览
- `Strategy`：净值、持仓、权重
- `Factors`：因子筛选、因子聚类、rolling IC
- `Experiments`：AI agent 自动实验结果
- `Agent Reports`：AI 研究报告

关闭：

```bash
lsof -i :8501
kill <PID>
```

## 公网分享

如果想把本地 dashboard 发给朋友，可以用 ngrok。

先启动 dashboard：

```bash
conda run -n quantforge streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

再启动 ngrok：

```bash
ngrok http http://127.0.0.1:8501
```

查看 ngrok 生成的网址：

```bash
curl http://127.0.0.1:4040/api/tunnels
```

关闭公网分享：

```bash
lsof -i :4040
kill <PID>
```

注意：

- ngrok 免费 URL 通常是临时的，重启后会变。
- 页面没有登录保护，不要展示 token、账户、持仓等敏感信息。
- Streamlit 本身已经是网页服务，通常不需要 Daphne。

## 常用完整流程

### ETF 策略研究

```bash
conda run -n quantforge python download_prices.py
conda run -n quantforge python run_market_regime.py
conda run -n quantforge python run_walkforward_factor_strategy.py --decorrelate-factors --report-dir reports/walkforward_factor_clustered_loose
conda run -n quantforge streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

### 主题研究

```bash
conda run -n quantforge python discover_theme_tickers.py --theme-name "uranium nuclear"
conda run -n quantforge python download_prices.py --universe data/themes/uranium_nuclear.yaml --start 2024-01-01 --output data/prices/uranium_nuclear_daily.parquet
conda run -n quantforge python run_theme_trends.py --theme data/themes/uranium_nuclear.yaml
conda run -n quantforge streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

### AI 自动实验

```bash
conda run -n quantforge python run_walkforward_factor_strategy.py --decorrelate-factors --report-dir reports/walkforward_factor_clustered_loose
conda run -n chunqiu python agent_tool_runner.py
conda run -n quantforge streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

## 报告文件

常见输出类型：

- `performance_summary.csv`：收益、波动、Sharpe、回撤
- `risk_summary.csv`：风险控制统计
- `turnover_summary.csv`：换手率和交易成本
- `holdings.csv`：再平衡持仓
- `daily_weights.csv`：每日权重
- `cumulative_wealth.csv`：累计净值
- `*.html`：Plotly 图表
- `*_summary.md`：文字总结

这些报告都在 `reports/` 下，dashboard 会自动读取其中一部分。

## 下一步可以扩展什么

- 机器学习横截面 ranking 策略
- 股票基本面数据
- 新闻 / 财报 / 宏观事件数据
- USDA 等非金融数据源接入
- 主题观察台的 LLM 自动选股增强
- 更严格的组合风险模型
- 交易成本、滑点、流动性约束
- 部署到云端并加登录保护
