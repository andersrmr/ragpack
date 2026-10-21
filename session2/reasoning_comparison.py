"""Compare reasoning effort on a fixed question set using the OpenAI API."""

import argparse
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from openai import APIError
from tinyagent import LLM

QUESTIONS = [
    {
        "id": "bat_ball",
        "question": (
            "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. "
            "How many cents does the ball cost? Reply with only the number."
        ),
        "answer": "5",
    },
    {
        "id": "machines",
        "question": (
            "If 5 machines take 5 minutes to make 5 widgets, how many minutes do 100 "
            "machines take to make 100 widgets? Reply with only the number."
        ),
        "answer": "5",
    },
    {
        "id": "lily_pads",
        "question": (
            "A patch of lily pads doubles in area every day. It covers the whole lake "
            "on day 48. On what day did it cover half the lake? Reply with only the day."
        ),
        "answer": "47",
    },
    {
        "id": "sequence",
        "question": (
            "What is the next number in this sequence: 2, 6, 12, 20, 30? "
            "Reply with only the number."
        ),
        "answer": "42",
    },
    {
        "id": "weekday",
        "question": (
            "If today is Tuesday, what day of the week will it be 100 days from today? "
            "Reply with only the weekday."
        ),
        "answer": "Thursday",
    },
]


def normalize(text: str) -> str:
    """Normalize case and surrounding punctuation for exact-answer grading."""
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def summarize(
    records: list[dict], input_price: float | None, output_price: float | None
) -> list[dict]:
    """Aggregate trial results by requested reasoning effort."""
    summaries = []
    for effort in dict.fromkeys(row["reasoning_effort"] for row in records):
        group = [row for row in records if row["reasoning_effort"] == effort]
        durations = [row["latency_seconds"] for row in group]
        prompt_tokens = [
            row["input_tokens"] for row in group if row["input_tokens"] is not None
        ]
        output_tokens = [
            row["output_tokens"] for row in group if row["output_tokens"] is not None
        ]
        reasoning_tokens = [
            row["reasoning_tokens"]
            for row in group
            if row["reasoning_tokens"] is not None
        ]
        summary = {
            "reasoning_effort": effort,
            "attempts": len(group),
            "correct": sum(row["correct"] for row in group),
            "accuracy": sum(row["correct"] for row in group) / len(group),
            "mean_latency_seconds": statistics.mean(durations),
            "median_latency_seconds": statistics.median(durations),
            "input_tokens": sum(prompt_tokens) if prompt_tokens else None,
            "output_tokens": sum(output_tokens) if output_tokens else None,
            "reasoning_tokens": sum(reasoning_tokens) if reasoning_tokens else None,
        }
        if (
            input_price is not None
            and output_price is not None
            and prompt_tokens
            and output_tokens
        ):
            summary["estimated_cost_usd"] = (
                summary["input_tokens"] * input_price
                + summary["output_tokens"] * output_price
            ) / 1_000_000
        else:
            summary["estimated_cost_usd"] = None
        summaries.append(summary)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Model ID; defaults to TINYAGENT_MODEL.")
    parser.add_argument(
        "--efforts",
        default="low,medium,high",
        help="Comma-separated reasoning efforts to compare (default: low,medium,high).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Trials per question and effort (default: 3).",
    )
    parser.add_argument(
        "--input-price", type=float, help="Current USD price per million input tokens."
    )
    parser.add_argument(
        "--output-price",
        type=float,
        help="Current USD price per million output tokens.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON results path (default: session2/results/<timestamp>.json).",
    )
    args = parser.parse_args()
    args.efforts = [
        effort.strip() for effort in args.efforts.split(",") if effort.strip()
    ]
    if not args.efforts:
        parser.error("provide at least one reasoning effort")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if (args.input_price is None) != (args.output_price is None):
        parser.error("provide both --input-price and --output-price, or neither")
    if any(
        price < 0
        for price in (args.input_price, args.output_price)
        if price is not None
    ):
        parser.error("prices cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    model = args.model
    results = []
    llm = None
    failures = []
    interrupted = False
    try:
        # Initialize once; update effort per condition to hold other settings constant.
        llm = LLM(model=model, reasoning_effort=args.efforts[0])
        model = llm.model
        for effort in args.efforts:
            llm.reasoning_effort = effort
            failed_effort = False
            for repetition in range(1, args.repetitions + 1):
                for item in QUESTIONS:
                    started = time.perf_counter()
                    try:
                        response = llm.generate(
                            [{"role": "user", "content": item["question"]}]
                        )
                    except APIError as error:
                        print(
                            f"Failed: {effort} / {item['id']} / trial {repetition}: {error}"
                        )
                        failures.append(
                            {
                                "reasoning_effort": effort,
                                "question_id": item["id"],
                                "trial": repetition,
                                "error": str(error),
                            }
                        )
                        failed_effort = True
                        break
                    latency = time.perf_counter() - started
                    correct = normalize(response.content) == normalize(item["answer"])
                    metadata = response.metadata or {}
                    row = {
                        "reasoning_effort": effort,
                        "question_id": item["id"],
                        "trial": repetition,
                        "answer_key": item["answer"],
                        "answer": response.content,
                        "correct": correct,
                        "latency_seconds": round(latency, 4),
                        "input_tokens": metadata.get("input_tokens"),
                        "output_tokens": metadata.get("output_tokens"),
                        "reasoning_tokens": metadata.get("reasoning_tokens"),
                        "reasoning_summary": response.reasoning,
                    }
                    results.append(row)
                    print(
                        f"{effort:>7} | {item['id']:<10} | trial {repetition} | "
                        f"{'correct' if correct else 'incorrect'} | {latency:.2f}s | "
                        f"in={row['input_tokens']} out={row['output_tokens']} "
                        f"reasoning={row['reasoning_tokens']}"
                    )
                if failed_effort:
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; saving completed trials.")
    except ValueError as error:
        print(f"Configuration error: {error}")
        return 1
    finally:
        if llm is not None:
            llm.close()

    if not results and not failures:
        print("No results to save.")
        return 1

    summaries = summarize(results, args.input_price, args.output_price)
    print("\nSummary")
    for row in summaries:
        print(
            f"{row['reasoning_effort']:>7} | accuracy={row['correct']}/{row['attempts']} "
            f"({row['accuracy']:.1%}) | mean={row['mean_latency_seconds']:.2f}s "
            f"| input={row['input_tokens']} output={row['output_tokens']} "
            f"reasoning={row['reasoning_tokens']} | cost={row['estimated_cost_usd']}"
        )

    now = datetime.now(ZoneInfo("America/New_York"))
    output = args.output or Path("results") / f"reasoning-{now:%Y%m%d-%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"Refusing to overwrite existing results file: {output}")
        return 1
    payload = {
        "model": model,
        "created_at": now.isoformat(),
        "questions": QUESTIONS,
        "configuration": {
            "reasoning_efforts": args.efforts,
            "repetitions": args.repetitions,
            "input_price_per_million_usd": args.input_price,
            "output_price_per_million_usd": args.output_price,
        },
        "summary": summaries,
        "trials": results,
        "failures": failures,
        "interrupted": interrupted,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"\nSaved results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
