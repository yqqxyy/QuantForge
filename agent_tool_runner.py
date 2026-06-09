from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from local_llm import DEFAULT_HF_HOME, DEFAULT_MODEL, LocalLLMConfig, LocalQwenMLX, build_chat_prompt
from research_agent import format_table, load_report_snapshot, strip_thinking_trace


DEFAULT_BASELINE_DIR = Path("reports/walkforward_factor_clustered_loose")
DEFAULT_OUTPUT_DIR = Path("reports/agent_tool_runs")
STRATEGY_SYMBOL = "WALKFORWARD_FACTOR_NET"

ALLOWED_PARAMS = {
    "decorrelate_factors",
    "factor_corr_threshold",
    "min_abs_ic",
    "min_directional_ic_fraction",
    "min_abs_ic_ir",
    "min_effective_months",
    "train_months",
    "min_train_months",
    "forward_days",
    "top_k",
    "target_volatility",
    "market_filter_symbol",
    "transaction_cost_bps",
}

FLOAT_PARAMS = {
    "factor_corr_threshold",
    "min_abs_ic",
    "min_directional_ic_fraction",
    "min_abs_ic_ir",
    "target_volatility",
    "transaction_cost_bps",
}
INT_PARAMS = {"min_effective_months", "train_months", "min_train_months", "forward_days", "top_k"}
BOOL_PARAMS = {"decorrelate_factors"}
STRING_PARAMS = {"market_filter_symbol"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate, validate, run, and compare AI-proposed quant experiments.")
    parser.add_argument("--baseline-report-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--max-tokens", type=int, default=1100)
    parser.add_argument("--max-experiments", type=int, default=3)
    parser.add_argument("--plan-only", action="store_true", help="Save the experiment plan but do not execute it.")
    parser.add_argument("--use-fallback-plan", action="store_true", help="Skip the LLM and use a built-in plan.")
    parser.add_argument("--skip-recap", action="store_true", help="Do not call the local model to summarize experiment results.")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_value(value: str) -> Any:
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_baseline_params(report_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(report_dir / "walkforward_config.csv")
    return {row["key"]: parse_value(row["value"]) for row in rows}


def load_strategy_metrics(report_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(report_dir / "performance_summary.csv")
    for row in rows:
        if row.get("symbol") == STRATEGY_SYMBOL:
            return {
                "symbol": row["symbol"],
                "annual_return": float(row["annual_return"]),
                "annual_volatility": float(row["annual_volatility"]),
                "sharpe": float(row["sharpe"]),
                "max_drawdown": float(row["max_drawdown"]),
                "calmar": float(row["calmar"]),
            }
    raise ValueError(f"{report_dir} does not contain {STRATEGY_SYMBOL}.")


def fallback_plan() -> list[dict[str, Any]]:
    return [
        {
            "name": "cluster_corr_070",
            "rationale": "Use a stricter clustering threshold to reduce redundant factor voting.",
            "params": {"factor_corr_threshold": 0.70},
        },
        {
            "name": "longer_training_window",
            "rationale": "Use a longer IC estimation window to reduce short-term noise.",
            "params": {"train_months": 48, "min_train_months": 24},
        },
        {
            "name": "vol_target_012",
            "rationale": "Test whether volatility targeting improves drawdown and risk-adjusted return.",
            "params": {"target_volatility": 0.12},
        },
    ]


def build_planner_prompt(snapshot: dict[str, str], baseline_params: dict[str, Any]) -> str:
    system_prompt = """
你是一个谨慎的量化研究规划 agent。你只能提出可由白名单参数表达的实验。
不要输出隐藏思考，不要输出 <think>。不要写 shell 命令。
只输出 JSON 数组，不要输出 Markdown，不要输出解释性段落。
"""
    user_prompt = f"""
/no_think

请基于以下量化研究报告，提出最多 3 个下一轮实验。
每个实验必须是 JSON object，格式严格如下：
[
  {{
    "name": "short_snake_case_name",
    "rationale": "中文一句话说明实验目的",
    "params": {{"factor_corr_threshold": 0.7}}
  }}
]

只能使用这些 params key：
{sorted(ALLOWED_PARAMS)}

baseline params:
{json.dumps(baseline_params, ensure_ascii=False, indent=2)}

策略表现：
{snapshot["strategy_metrics"]}

基准表现：
{snapshot["benchmark_metrics"]}

因子稳定性和聚类摘要：
{snapshot["factor_selection"]}

风险状态：
{snapshot["risk"]}
"""
    return build_chat_prompt(system_prompt, user_prompt)


def extract_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = strip_thinking_trace(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, list):
        raise ValueError("Planner output must be a JSON array.")
    return parsed


def sanitize_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    return cleaned or fallback


def coerce_param(key: str, value: Any) -> Any:
    if key not in ALLOWED_PARAMS:
        raise ValueError(f"Unsupported param: {key}")
    if value in {"", "none", "None", "null"}:
        value = None

    if key in FLOAT_PARAMS:
        return None if value is None else float(value)
    if key in INT_PARAMS:
        return int(value)
    if key in BOOL_PARAMS:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes"}
        return bool(value)
    if key in STRING_PARAMS:
        return None if value is None else str(value).upper()
    return value


def validate_param_ranges(params: dict[str, Any]) -> None:
    bounded = {
        "factor_corr_threshold": (0.0, 1.0),
        "min_directional_ic_fraction": (0.0, 1.0),
        "min_abs_ic": (0.0, 1.0),
        "min_abs_ic_ir": (0.0, 10.0),
        "target_volatility": (0.01, 1.0),
        "transaction_cost_bps": (0.0, 100.0),
    }
    for key, (lower, upper) in bounded.items():
        value = params.get(key)
        if value is not None and not lower <= float(value) <= upper:
            raise ValueError(f"{key}={value} is outside [{lower}, {upper}].")

    for key in INT_PARAMS:
        value = params.get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"{key} must be positive.")

    if params.get("market_filter_symbol") not in {None, "SPY"}:
        raise ValueError("Only market_filter_symbol=SPY is currently allowed.")


def validate_plan(plan: list[dict[str, Any]], max_experiments: int) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, item in enumerate(plan[:max_experiments], start=1):
        if not isinstance(item, dict):
            raise ValueError("Each experiment must be an object.")
        name = sanitize_name(str(item.get("name", "")), f"experiment_{index}")
        while name in seen_names:
            name = f"{name}_{index}"
        seen_names.add(name)

        raw_params = item.get("params", {})
        if not isinstance(raw_params, dict):
            raise ValueError(f"{name}: params must be an object.")
        params = {}
        rejected_params = {}
        for key, value in raw_params.items():
            if key not in ALLOWED_PARAMS:
                rejected_params[key] = value
                continue
            params[key] = coerce_param(key, value)
        validate_param_ranges(params)
        experiments.append(
            {
                "name": name,
                "rationale": str(item.get("rationale", "")).strip(),
                "params": params,
                "rejected_params": rejected_params,
            }
        )

    if not experiments:
        raise ValueError("No valid experiments proposed.")
    return experiments


def merge_params(baseline: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(baseline)
    merged.update(overrides)
    merged["decorrelate_factors"] = bool(merged.get("decorrelate_factors", True))
    return merged


def build_command(params: dict[str, Any], report_dir: Path) -> list[str]:
    command = [
        "conda",
        "run",
        "-n",
        "quantforge",
        "python",
        "run_walkforward_factor_strategy.py",
        "--report-dir",
        str(report_dir),
    ]
    if params.get("decorrelate_factors"):
        command.append("--decorrelate-factors")

    flag_map = {
        "factor_corr_threshold": "--factor-corr-threshold",
        "min_abs_ic": "--min-abs-ic",
        "min_directional_ic_fraction": "--min-directional-ic-fraction",
        "min_abs_ic_ir": "--min-abs-ic-ir",
        "min_effective_months": "--min-effective-months",
        "train_months": "--train-months",
        "min_train_months": "--min-train-months",
        "forward_days": "--forward-days",
        "top_k": "--top-k",
        "target_volatility": "--target-volatility",
        "market_filter_symbol": "--market-filter-symbol",
        "transaction_cost_bps": "--transaction-cost-bps",
    }
    for key, flag in flag_map.items():
        value = params.get(key)
        if value is not None:
            command.extend([flag, str(value)])
    return command


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    logging.info("Running: %s", " ".join(command))
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)


def percent(value: float) -> str:
    return f"{value:.2%}"


def write_summary_markdown(
    path: Path,
    baseline_dir: Path,
    baseline_metrics: dict[str, Any],
    result_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Agent Tool Runner Summary",
        "",
        f"Baseline report: `{baseline_dir}`",
        "",
        "## Baseline",
        "",
        (
            f"- annual_return={percent(baseline_metrics['annual_return'])}, "
            f"vol={percent(baseline_metrics['annual_volatility'])}, "
            f"sharpe={baseline_metrics['sharpe']:.3f}, "
            f"max_drawdown={percent(baseline_metrics['max_drawdown'])}, "
            f"calmar={baseline_metrics['calmar']:.3f}"
        ),
        "",
        "## Experiments",
        "",
    ]
    for row in result_rows:
        lines.extend(
            [
                f"### {row['name']}",
                "",
                f"- rationale: {row['rationale']}",
                f"- report_dir: `{row['report_dir']}`",
                f"- command: `{' '.join(row['command'])}`",
                (
                    f"- metrics: annual_return={percent(row['annual_return'])}, "
                    f"vol={percent(row['annual_volatility'])}, "
                    f"sharpe={row['sharpe']:.3f}, "
                    f"max_drawdown={percent(row['max_drawdown'])}, "
                    f"calmar={row['calmar']:.3f}"
                ),
                (
                    f"- delta: annual_return={row['delta_annual_return']:+.3%}, "
                    f"sharpe={row['delta_sharpe']:+.3f}, "
                    f"max_drawdown={row['delta_max_drawdown']:+.3%}, "
                    f"calmar={row['delta_calmar']:+.3f}"
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_plan(args: argparse.Namespace, baseline_params: dict[str, Any], snapshot: dict[str, str], run_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    if args.use_fallback_plan:
        prompt = "(fallback plan used; local LLM was not called)"
        raw_plan = fallback_plan()
    else:
        prompt = build_planner_prompt(snapshot, baseline_params)
        llm = LocalQwenMLX(
            LocalLLMConfig(
                model=args.model,
                hf_home=args.hf_home,
                max_tokens=args.max_tokens,
            )
        )
        response = strip_thinking_trace(llm.generate(prompt))
        (run_dir / "planner_response.txt").write_text(response, encoding="utf-8")
        raw_plan = extract_json_array(response)

    plan = validate_plan(raw_plan, args.max_experiments)
    return prompt, plan


def build_recap_prompt(summary_markdown: str, results_rows: list[dict[str, Any]]) -> str:
    system_prompt = """
你是一个谨慎的量化研究复盘 agent。你只根据实验结果表做判断，不给实盘投资建议。
不要输出隐藏思考，不要输出 <think>。
输出要短、具体、工程化，重点是：哪些实验保留、哪些拒绝、下一步怎么做。
"""
    result_table_rows = []
    for row in results_rows:
        result_table_rows.append(
            {
                "name": row["name"],
                "annual_return": f"{row['annual_return']:.6f}",
                "sharpe": f"{row['sharpe']:.6f}",
                "max_drawdown": f"{row['max_drawdown']:.6f}",
                "calmar": f"{row['calmar']:.6f}",
                "delta_annual_return": f"{row['delta_annual_return']:.6f}",
                "delta_sharpe": f"{row['delta_sharpe']:.6f}",
                "delta_max_drawdown": f"{row['delta_max_drawdown']:.6f}",
                "delta_calmar": f"{row['delta_calmar']:.6f}",
            }
        )
    result_table = format_table(
        result_table_rows,
        [
            "name",
            "annual_return",
            "sharpe",
            "max_drawdown",
            "calmar",
            "delta_annual_return",
            "delta_sharpe",
            "delta_max_drawdown",
            "delta_calmar",
        ],
        max_rows=len(result_table_rows),
    )
    user_prompt = f"""
/no_think

下面是 agent 自动提出并执行的一轮量化实验结果。

数值摘要：
{summary_markdown}

结果表：
{result_table}

请用中文输出：
1. 本轮实验最重要的结论，用 1-2 句话。
2. 每个实验是接受、拒绝还是观察，给一句理由。
3. 当前应保留的 research baseline。
4. 下一轮建议最多 3 个实验方向，只说方向，不要写命令。
5. 明确指出这轮实验没有证明什么。
"""
    return build_chat_prompt(system_prompt, user_prompt)


def deterministic_recap(baseline_metrics: dict[str, Any], result_rows: list[dict[str, Any]]) -> str:
    best = max(result_rows, key=lambda row: row["sharpe"])
    lines = [
        "# Agent Experiment Recap",
        "",
        "## 结论",
        "",
        (
            f"本轮最佳实验是 `{best['name']}`，Sharpe={best['sharpe']:.3f}，"
            f"相对 baseline 变化 {best['delta_sharpe']:+.3f}。"
        ),
        (
            f"Baseline Sharpe={baseline_metrics['sharpe']:.3f}，"
            f"年化收益={percent(baseline_metrics['annual_return'])}，"
            f"最大回撤={percent(baseline_metrics['max_drawdown'])}。"
        ),
        "",
        "## 实验判断",
        "",
    ]
    for row in result_rows:
        if row["delta_sharpe"] > 0.01 and row["delta_max_drawdown"] >= -0.02:
            decision = "接受"
        elif row["delta_sharpe"] < -0.02 or row["delta_calmar"] < -0.05:
            decision = "拒绝"
        else:
            decision = "观察"
        lines.append(
            f"- `{row['name']}`: {decision}。"
            f"年化收益 {percent(row['annual_return'])} ({row['delta_annual_return']:+.3%})，"
            f"Sharpe {row['sharpe']:.3f} ({row['delta_sharpe']:+.3f})，"
            f"最大回撤 {percent(row['max_drawdown'])} ({row['delta_max_drawdown']:+.3%})。"
        )

    if best["delta_sharpe"] > 0.01:
        baseline = best["report_dir"]
    else:
        baseline = "reports/walkforward_factor_clustered_loose"

    lines.extend(
        [
            "",
            "## 当前 Baseline",
            "",
            f"建议继续保留 `{baseline}` 作为 research baseline。",
            "",
            "## 下一轮方向",
            "",
            "- 做交易成本敏感性测试，确认高换手版本是否仍稳健。",
            "- 测试加入 `SPY` 市场过滤后是否能改善回撤和 Calmar。",
            "- 对训练窗口长度做 24/36/48/60 个月的小网格测试。",
            "",
            "## 没有证明什么",
            "",
            "本轮实验只说明这些参数在当前 ETF 池和历史样本里表现如何；它没有证明策略未来有效，也没有证明因子有稳定长期 alpha。",
        ]
    )
    return "\n".join(lines)


def recap_is_useful(response: str) -> bool:
    cleaned = response.strip()
    if len(cleaned) < 220:
        return False
    required_terms = ["接受", "拒绝", "观察", "baseline"]
    return any(term in cleaned for term in required_terms) and "本轮实验" in cleaned


def generate_recap(args: argparse.Namespace, run_dir: Path, baseline_metrics: dict[str, Any], result_rows: list[dict[str, Any]]) -> None:
    summary_path = run_dir / "summary.md"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary for recap: {summary_path}")
    summary_markdown = summary_path.read_text(encoding="utf-8")
    prompt = build_recap_prompt(summary_markdown, result_rows)
    (run_dir / "recap_prompt.txt").write_text(prompt, encoding="utf-8")

    llm = LocalQwenMLX(
        LocalLLMConfig(
            model=args.model,
            hf_home=args.hf_home,
            max_tokens=args.max_tokens,
        )
    )
    response = strip_thinking_trace(llm.generate(prompt))
    if not recap_is_useful(response):
        response = deterministic_recap(baseline_metrics, result_rows)
    (run_dir / "agent_recap.md").write_text(response, encoding="utf-8")
    print(response)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = args.output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        baseline_params = load_baseline_params(args.baseline_report_dir)
        baseline_metrics = load_strategy_metrics(args.baseline_report_dir)
        snapshot = load_report_snapshot(args.baseline_report_dir)
        prompt, plan = generate_plan(args, baseline_params, snapshot, run_dir)

        (run_dir / "planner_prompt.txt").write_text(prompt, encoding="utf-8")
        (run_dir / "validated_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

        result_rows: list[dict[str, Any]] = []
        if not args.plan_only:
            for experiment in plan:
                params = merge_params(baseline_params, experiment["params"])
                report_dir = args.output_dir / "experiment_reports" / timestamp / experiment["name"]
                command = build_command(params, report_dir)
                completed = run_command(command, Path.cwd())

                log_dir = run_dir / "logs"
                log_dir.mkdir(exist_ok=True)
                (log_dir / f"{experiment['name']}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
                (log_dir / f"{experiment['name']}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
                if completed.returncode != 0:
                    raise RuntimeError(f"{experiment['name']} failed with exit code {completed.returncode}.")

                metrics = load_strategy_metrics(report_dir)
                result_rows.append(
                    {
                        "name": experiment["name"],
                        "rationale": experiment["rationale"],
                        "report_dir": str(report_dir),
                        "command": command,
                        **metrics,
                        "delta_annual_return": metrics["annual_return"] - baseline_metrics["annual_return"],
                        "delta_annual_volatility": metrics["annual_volatility"] - baseline_metrics["annual_volatility"],
                        "delta_sharpe": metrics["sharpe"] - baseline_metrics["sharpe"],
                        "delta_max_drawdown": metrics["max_drawdown"] - baseline_metrics["max_drawdown"],
                        "delta_calmar": metrics["calmar"] - baseline_metrics["calmar"],
                    }
                )

            csv_rows = [{**row, "command": " ".join(row["command"])} for row in result_rows]
            write_csv_rows(run_dir / "experiment_results.csv", csv_rows)
            write_summary_markdown(run_dir / "summary.md", args.baseline_report_dir, baseline_metrics, result_rows)
            if not args.skip_recap:
                generate_recap(args, run_dir, baseline_metrics, result_rows)

        metadata = {
            "baseline_report_dir": str(args.baseline_report_dir),
            "output_dir": str(args.output_dir),
            "model": args.model,
            "hf_home": str(args.hf_home),
            "max_tokens": args.max_tokens,
            "max_experiments": args.max_experiments,
            "plan_only": args.plan_only,
            "use_fallback_plan": args.use_fallback_plan,
            "skip_recap": args.skip_recap,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("Saved tool-runner output to %s", run_dir)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
