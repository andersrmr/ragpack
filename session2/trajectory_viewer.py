"""Render a reasoning experiment or TinyAgent trajectory as local HTML."""

import argparse
import html
import json
from pathlib import Path


STYLE = """
<style>
body { max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #182230;
       background: #f7f8fa; font: 15px/1.5 system-ui, sans-serif; }
h1, h2 { line-height: 1.2; }
.meta, .card { background: white; border: 1px solid #d8dee8; border-radius: 8px;
               padding: 1rem; margin: 1rem 0; }
.meta { display: flex; gap: 2rem; flex-wrap: wrap; }
table { width: 100%; border-collapse: collapse; background: white; }
th, td { border: 1px solid #d8dee8; padding: .55rem; text-align: left; }
th { background: #edf1f6; }
details { background: white; border: 1px solid #d8dee8; border-radius: 6px;
          margin: .6rem 0; padding: .75rem; }
summary { cursor: pointer; font-weight: 650; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f1f3f6;
      border-radius: 5px; padding: .75rem; }
.label { color: #526071; font-size: .85rem; font-weight: 650; margin: .8rem 0 .2rem; }
.error { color: #a32020; }
</style>
"""


def escaped(value) -> str:
    return html.escape(str(value if value is not None else ""))


def render_summary(rows: list[dict]) -> str:
    if not rows:
        return "<p>No aggregate summary is available in this file.</p>"
    headings = (
        "Reasoning effort",
        "Attempts",
        "Accuracy",
        "Mean latency",
        "Input tokens",
        "Output tokens",
        "Reasoning tokens",
        "Est. cost (USD)",
    )
    output = ["<table><thead><tr>"]
    output.extend(f"<th>{escaped(label)}</th>" for label in headings)
    output.append("</tr></thead><tbody>")
    for row in rows:
        accuracy = row.get("accuracy")
        latency = row.get("mean_latency_seconds")
        cost = row.get("estimated_cost_usd")
        values = (
            row.get("reasoning_effort", ""),
            row.get("attempts", ""),
            f"{accuracy:.1%}" if isinstance(accuracy, (float, int)) else "",
            f"{latency:.2f}s" if isinstance(latency, (float, int)) else "",
            row.get("input_tokens", ""),
            row.get("output_tokens", ""),
            row.get("reasoning_tokens", ""),
            f"{cost:.6f}" if isinstance(cost, (float, int)) else "",
        )
        output.append(
            "<tr>" + "".join(f"<td>{escaped(value)}</td>" for value in values) + "</tr>"
        )
    output.append("</tbody></table>")
    return "".join(output)


def render_trial(trial: dict, questions: dict[str, dict]) -> str:
    question = questions.get(trial.get("question_id"), {}).get("question", "")
    tokens = {
        "input_tokens": trial.get("input_tokens"),
        "output_tokens": trial.get("output_tokens"),
        "reasoning_tokens": trial.get("reasoning_tokens"),
    }
    title = (
        f"{trial.get('reasoning_effort', 'unknown')} · "
        f"{trial.get('question_id', 'question')} · trial {trial.get('trial', '?')} · "
        f"{'correct' if trial.get('correct') else 'incorrect'} · "
        f"{trial.get('latency_seconds', '?')}s"
    )
    details = [f"<details><summary>{escaped(title)}</summary>"]
    details.append(f"<div class='label'>Question</div><pre>{escaped(question)}</pre>")
    details.append(
        f"<div class='label'>Answer</div><pre>{escaped(trial.get('answer'))}</pre>"
    )
    details.append(
        "<div class='label'>Answer key</div>"
        f"<pre>{escaped(trial.get('answer_key'))}</pre>"
    )
    details.append(
        "<div class='label'>Token usage</div>"
        f"<pre>{escaped(json.dumps(tokens, indent=2))}</pre>"
    )
    if trial.get("reasoning_summary"):
        details.append(
            f"<div class='label'>Reasoning summary</div>"
            f"<pre>{escaped(trial['reasoning_summary'])}</pre>"
        )
    details.append("</details>")
    return "".join(details)


def render_run(run: dict) -> str:
    metadata = run.get("metadata") or {}
    effort = metadata.get("reasoning_effort", "")
    question_id = metadata.get("question_id", "")
    trial = metadata.get("trial", "")
    correctness = metadata.get("correct")
    if correctness is True:
        accuracy = "correct"
    elif correctness is False:
        accuracy = "incorrect"
    else:
        accuracy = "failed"
    details = [
        f"<details><summary>Run {escaped(run.get('run_id'))} · "
        f"{escaped(effort)} · {escaped(question_id)} · trial {escaped(trial)} · "
        f"{accuracy} · "
        f"{escaped(run.get('duration_seconds'))}s</summary>"
    ]
    details.append(
        f"<div class='label'>Query</div><pre>{escaped(run.get('query'))}</pre>"
    )
    if metadata.get("answer_key") is not None:
        details.append(
            "<div class='label'>Answer key</div>"
            f"<pre>{escaped(metadata['answer_key'])}</pre>"
        )
    for step in run.get("steps", []):
        details.append(
            f"<div class='label'>Step {escaped(step.get('step_id'))}: "
            f"{escaped(step.get('model'))} at "
            f"{escaped(step.get('reasoning_effort'))} effort, "
            f"{escaped(step.get('latency_seconds'))}s</div>"
        )
        messages = json.dumps(
            step.get("input_messages", []), ensure_ascii=False, indent=2
        )
        details.append(
            f"<div class='label'>Input messages</div><pre>{escaped(messages)}</pre>"
        )
        usage = json.dumps(step.get("metadata") or {}, ensure_ascii=False, indent=2)
        details.append(
            f"<div class='label'>Token usage</div><pre>{escaped(usage)}</pre>"
        )
        if step.get("answer"):
            details.append(
                f"<div class='label'>Answer</div><pre>{escaped(step['answer'])}</pre>"
            )
        if step.get("reasoning"):
            details.append(
                "<div class='label'>Reasoning summary</div>"
                f"<pre>{escaped(step['reasoning'])}</pre>"
            )
        if step.get("error"):
            error = json.dumps(step["error"], ensure_ascii=False, indent=2)
            details.append(
                "<div class='label error'>Call error</div>"
                f"<pre class='error'>{escaped(error)}</pre>"
            )
    details.append("</details>")
    return "".join(details)


def render_report(document: dict) -> str:
    """Build a self-contained report from current or earlier experiment JSON."""
    questions = {item.get("id"): item for item in document.get("questions", [])}
    trials = document.get("trials") or []
    trajectory = document.get("trajectory") or {}
    runs = trajectory.get("runs") or []
    model = document.get("model", "unknown model")
    created_at = document.get("created_at", "unknown time")
    content = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>TinyAgent trajectory · {escaped(model)}</title>",
        STYLE,
        "</head><body>",
        "<h1>TinyAgent trajectory</h1>",
        "<section class='meta'>",
        f"<div><strong>Model</strong><br>{escaped(model)}</div>",
        f"<div><strong>Created</strong><br>{escaped(created_at)}</div>",
        f"<div><strong>Runs</strong><br>{len(runs) if runs else len(trials)}</div>",
        "</section>",
        "<h2>Summary by reasoning effort</h2>",
        render_summary(document.get("summary") or []),
    ]
    if runs:
        content.append("<h2>Agent runs and model calls</h2>")
        content.extend(render_run(run) for run in runs)
    elif trials:
        content.append("<h2>Recorded calls</h2>")
        content.extend(render_trial(trial, questions) for trial in trials)
    else:
        content.append("<p>No successful calls are recorded in this file.</p>")

    failures = document.get("failures") or []
    if failures and not runs:
        content.append("<h2>Failed calls</h2>")
        for failure in failures:
            title = (
                f"{failure.get('reasoning_effort', 'unknown')} · "
                f"{failure.get('question_id', 'question')} · "
                f"trial {failure.get('trial', '?')}"
            )
            content.append(
                f"<details><summary class='error'>{escaped(title)}</summary>"
                f"<pre class='error'>{escaped(failure.get('error'))}</pre></details>"
            )
    content.append("</body></html>")
    return "".join(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Experiment JSON file to inspect.")
    parser.add_argument(
        "--output", type=Path, help="HTML output path (defaults beside the JSON file)."
    )
    args = parser.parse_args()
    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read JSON input: {error}")
    if not isinstance(document, dict):
        parser.error("input must contain a JSON object")

    output = args.output or args.input.with_suffix(".html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(document), encoding="utf-8")
    print(f"Trajectory report saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
