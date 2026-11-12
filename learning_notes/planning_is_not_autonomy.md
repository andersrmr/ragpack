# Planning Is Not Autonomy: ReAct, Plan-and-Solve, and Explicit Planning State

**Course:** Introduction to Agents — Session 4  
**Themes:** ReAct, planning, Plan-and-Solve, state management, adaptive execution, planning overhead

## The central insight

**ReAct and Plan-and-Solve address related but distinct aspects of agentic behavior: ReAct provides the mechanism for iterative decision-making and execution, while Plan-and-Solve introduces an explicit representation of the work the agent intends to perform.**

An autonomous agent does not necessarily need to formulate a plan before acting. It can select an initial action, observe the result, and determine the next step dynamically.

Explicit planning becomes valuable when the task involves multiple dependencies, substantial coordination, or a need to track progress across many actions.

**Planning helps organize agent behavior; the autonomous loop is what makes successive model-directed actions executable.**

## 1. ReAct: decide, act, observe, repeat

ReAct combines reasoning and acting in a feedback-driven loop:

**Thought → Action → Observation → Updated decision**

At each iteration, the model evaluates the available context and decides whether to request another tool or provide a final answer.

The model may implicitly anticipate several future steps, but it does not necessarily maintain an explicit, persistent task plan.

For example, when asked to calculate a multistep arithmetic expression, a ReAct agent might request addition, observe the result, request multiplication, observe that result, and then request subtraction.

The sequence emerges through successive model decisions.

With native reasoning and tool calling, the course's `NativeReAct` class requires very little additional machinery. The primary explicit configuration is `max_steps`, which limits execution.

This illustrates that the **essential architecture of ReAct is the feedback loop, not a particular prompting format or planning document**.

## 2. Plan-and-Solve: making the plan explicit

Plan-and-Solve adds tools that allow the model to create, update, and inspect a structured plan.

In the Session 4 implementation, the principal tools are:

- `update_plan(tasks)`: Creates or replaces a numbered checklist of intended tasks.
- `mark_done(index)`: Marks a specified task as completed.

For a shopping calculation, the plan might look like:

```text
[ ] 1. Calculate the notebook subtotal.
[ ] 2. Calculate the pen subtotal.
[ ] 3. Combine the subtotals.
[ ] 4. Apply sales tax.
[ ] 5. Report the final amount.
```

As the agent executes the necessary operations, it updates the checklist.

The plan is no longer merely an idea implicit in the model's reasoning. It becomes **structured application state that the harness can preserve, expose to the model, and render for inspection**.

In the course implementation, plan changes are returned as tool observations and become part of the conversation history, allowing subsequent model requests to see the plan's current state.

This introduces a new form of coordination between the model and its harness.

## 3. Why explicit planning matters

A persistent plan can provide several advantages:

| Capability | ReAct without explicit planning | ReAct with explicit planning |
|---|---|---|
| Next-action selection | Based on current context and observations | Based on current context, observations, and plan state |
| Progress tracking | Inferred from conversation history | Represented explicitly in a checklist |
| Task dependencies | Managed implicitly by model decisions | Can be represented as ordered steps |
| Replanning | Model changes its next action | Model can revise the documented plan |
| Observability | Actions and observations are visible | Actions, observations, and intended progress are visible |

The strongest benefit appears in longer or more complex tasks, where an agent must coordinate several operations without losing track of what has already been completed.

Explicit plans can also improve human oversight: a reviewer can inspect the intended work before or during execution.

However, **a plan does not guarantee that the model will follow it correctly**, nor does marking a task complete prove that the underlying work succeeded.

The harness may need additional validation to distinguish declared progress from verified progress.

## 4. An airport operations example: investigating passenger impacts

Suppose an agent is asked:

*"Were yesterday's flight cancellations unusually high, and did they affect passenger volumes?"*

A ReAct agent might begin by retrieving cancellation counts. After observing an anomaly, it could request historical comparisons, then passenger-volume information, and progressively develop its analysis.

A Plan-and-Solve agent might first establish an explicit investigation plan:

1. Retrieve cancellation counts for the specified day.
2. Establish an appropriate historical baseline.
3. Compare observed cancellations against that baseline.
4. Retrieve passenger-volume observations and comparable reference periods.
5. Examine temporal relationships between cancellations and passenger volumes.
6. Summarize findings, limitations, and remaining uncertainty.

Both approaches can use the same tools and autonomous execution loop.

The difference is that the planned version makes its intended analytical sequence explicit and trackable.

Observations may still require adaptation. For example, discovering incomplete passenger-volume data might lead the model to revise the plan and investigate data quality before drawing conclusions.

Importantly, even an explicit plan does not establish causality. The agent must distinguish temporal association from evidence that cancellations caused a change in passenger volumes.

## 5. Planning introduces overhead and new failure modes

Planning is not automatically beneficial.

The course observes that the Plan-and-Solve arithmetic demonstration takes substantially more steps than the corresponding unplanned calculation—approximately ten versus five in the example.

Planning may increase:

- Model requests, token consumption, and latency.
- Tool calls devoted to maintaining the plan.
- Opportunities for inconsistent bookkeeping.
- Complexity in handling revisions and incomplete work.

The course also notes that smaller models sometimes skip completion updates or confuse task numbering.

Another limitation is that the course's `update_plan()` replaces the checklist and resets completion markers. This is acceptable for an educational example but potentially problematic in longer-running systems where earlier accomplishments should not be lost during replanning.

More robust planning mechanisms might preserve completed work, distinguish pending from blocked tasks, record supporting evidence, and track plan revisions.

The appropriate level of planning should therefore reflect the task's complexity and coordination requirements.

## 6. Planning and autonomy are independent architectural choices

A system can maintain an explicit plan while requiring human approval after every action. Such a system has planning capabilities but limited execution autonomy.

Conversely, an autonomous ReAct agent can repeatedly select and execute actions without maintaining an explicit checklist.

This distinction is especially relevant to the TinyAgent progression:

- **Session 3:** Model-directed tool selection, with manual continuation.
- **Session 4 Stage A:** Autonomous ReAct, without an explicit plan.
- **Session 4 Stage B:** Autonomous ReAct with structured planning tools.

The staged implementation makes it possible to isolate and evaluate the contribution of each capability.

It also suggests a useful experiment: compare the same tasks with and without explicit planning, measuring tool selection, completion quality, step counts, latency, and token consumption.

## Takeaway

> **ReAct governs the iterative process of deciding, acting, and observing. Plan-and-Solve adds explicit state representing the intended work. Planning can improve coherence, transparency, and progress tracking, but it does not itself create autonomy—and autonomy does not require an explicit plan. The appropriate planning strategy depends on task complexity, reliability, and cost.**

**Connection to other notes:** `the_agentic_loop_and_react.md` explains the feedback mechanism underlying both approaches; `rag_vs_model_directed_retrieval.md` illustrates model-directed information acquisition; the Session 4 harness note explains how autonomous execution is orchestrated.

*Provenance: Based on Session 4 course Sections 3–4, including `NativeReAct`, `NativePlanAndSolve`, `update_plan`, and `mark_done`.*