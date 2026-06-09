# conda run -n chunqiu python research_agent.py --max-tokens 1800
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from local_llm import DEFAULT_HF_HOME, DEFAULT_MODEL, LocalLLMConfig, LocalQwenMLX, build_chat_prompt


DEFAULT_REPORT_DIR = Path("reports/walkforward_factor_clustered_loose")
DEFAULT_OUTPUT_DIR = Path("reports/agent_experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use a local Qwen model as a quant research agent.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and save the prompt without calling the local model.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing report file: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def format_table(rows: list[dict[str, str]], columns: list[str], max_rows: int = 8) -> str:
    if not rows:
        return "(empty)"
    available = [column for column in columns if column in rows[0]]
    if not available:
        return "(no matching columns)"

    selected_rows = rows[:max_rows]
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in selected_rows))
        for column in available
    }
    header = " ".join(column.rjust(widths[column]) for column in available)
    body = [
        " ".join(str(row.get(column, "")).rjust(widths[column]) for column in available)
        for row in selected_rows
    ]
    return "\n".join([header, *body])


def rows_to_table(rows: list[dict[str, str]], max_rows: int = 20) -> str:
    if not rows:
        return "(empty)"
    return format_table(rows, list(rows[0]), max_rows=max_rows)


def load_report_snapshot(report_dir: Path) -> dict[str, str]:
    performance = read_csv_rows(report_dir / "performance_summary.csv")
    factor_selection = read_csv_rows(report_dir / "factor_selection_summary.csv")
    holdings = read_csv_rows(report_dir / "holdings.csv")
    turnover = read_csv_rows(report_dir / "turnover_summary.csv")
    risk = read_csv_rows(report_dir / "risk_summary.csv")
    config = read_csv_rows(report_dir / "walkforward_config.csv")

    strategy = [row for row in performance if row.get("symbol") == "WALKFORWARD_FACTOR_NET"]
    if not strategy:
        raise ValueError("performance_summary.csv does not contain WALKFORWARD_FACTOR_NET.")
    benchmarks = [row for row in performance if row.get("symbol") != "WALKFORWARD_FACTOR_NET"]

    return {
        "strategy_metrics": format_table(
            strategy,
            ["symbol", "annual_return", "annual_volatility", "sharpe", "max_drawdown", "calmar"],
            max_rows=1,
        ),
        "benchmark_metrics": format_table(
            benchmarks,
            ["symbol", "annual_return", "annual_volatility", "sharpe", "max_drawdown", "calmar"],
        ),
        "factor_selection": format_table(
            factor_selection,
            [
                "factor",
                "selection_rate",
                "pre_cluster_selection_rate",
                "cluster_suppression_rate",
                "average_abs_ic",
                "average_abs_ic_ir",
                "last_selected",
            ],
        ),
        "latest_holdings": rows_to_table(holdings[-12:]),
        "turnover": rows_to_table(turnover),
        "risk": rows_to_table(risk),
        "config": rows_to_table(config),
    }


def build_research_prompt(snapshot: dict[str, str], report_dir: Path) -> str:
    system_prompt = """
你是一个谨慎的量化研究助理。你不会声称策略一定赚钱，也不会给实盘投资建议。
你的任务是阅读研究报告，诊断策略弱点，提出下一轮可复现实验。
输出要具体、结构化、简洁，并且所有建议都要能通过现有 Python 脚本验证。
不要输出隐藏思考过程，不要输出 <think> 标签。
只能使用用户给你的表格数字，不要编造表格里没有的数值。
"""

    user_prompt = f"""
/no_think

请分析这个量化 ETF 项目的最新 walk-forward 因子聚类策略报告。

报告目录：
{report_dir}

策略表现：
{snapshot["strategy_metrics"]}

基准表现：
{snapshot["benchmark_metrics"]}

因子稳定性和聚类摘要：
{snapshot["factor_selection"]}

最近持仓：
{snapshot["latest_holdings"]}

换手：
{snapshot["turnover"]}

风险状态：
{snapshot["risk"]}

策略配置：
{snapshot["config"]}

可用实验命令模板：
conda run -n quantforge python run_walkforward_factor_strategy.py --decorrelate-factors --report-dir reports/<new_report_dir>

可调整参数包括：
--factor-corr-threshold
--min-abs-ic
--min-directional-ic-fraction
--min-abs-ic-ir
--train-months
--min-train-months
--forward-days
--top-k
--target-volatility
--market-filter-symbol SPY

请用中文输出：
1. 当前策略的一句话评价。
2. 它相对 QQQ、SPY、等权组合的主要优缺点。
3. 因子层面的观察，尤其是哪些因子可能重复、哪些因子较稳定。
4. 过拟合风险在哪里。
5. 下一轮最值得跑的 3 个实验，每个实验必须给出一条完整可复制的 shell 命令，不要给伪代码。
6. 你建议优先保留哪个版本作为当前 research baseline。
"""
    return build_chat_prompt(system_prompt, user_prompt)


def strip_thinking_trace(response: str) -> str:
    if "</think>" in response:
        return response.split("</think>", 1)[1].strip()
    return response.strip()


def save_agent_run(output_dir: Path, prompt: str, response: str | None, metadata: dict[str, object]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if response is not None:
        (run_dir / "agent_report.md").write_text(response, encoding="utf-8")
    return run_dir


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    try:
        snapshot = load_report_snapshot(args.report_dir)
        prompt = build_research_prompt(snapshot, args.report_dir)
        metadata = {
            "report_dir": str(args.report_dir),
            "model": args.model,
            "hf_home": str(args.hf_home),
            "max_tokens": args.max_tokens,
            "dry_run": args.dry_run,
        }

        response: str | None = None
        if not args.dry_run:
            llm = LocalQwenMLX(
                LocalLLMConfig(
                    model=args.model,
                    hf_home=args.hf_home,
                    max_tokens=args.max_tokens,
                )
            )
            response = strip_thinking_trace(llm.generate(prompt))
            print(response)

        run_dir = save_agent_run(args.output_dir, prompt, response, metadata)
        logging.info("Saved agent run to %s", run_dir)
        if args.dry_run:
            logging.info("Dry run only. Prompt saved; local model was not called.")
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
