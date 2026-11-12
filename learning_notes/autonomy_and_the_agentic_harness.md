# Autonomy and the Agentic Harness: From Tool Calls to Self-Continuing Execution

**Course:** Introduction to Agents — Session 4  
**Themes:** Autonomy, harness architecture, ReAct, control flow, execution boundaries, stopping conditions

## The central insight

**The defining transition from a tool-using assistant to an autonomous agent occurs when the harness automatically manages successive cycles of model decision-making, tool execution, and observation.**

In Session 3, TinyAgent could request and execute a tool, return the result to conversation memory, and record the interaction in its trajectory. However, the user still determined when to advance the process by pressing **Continue**.

Session 4 introduces an automated execution loop. Following each tool observation, the harness makes another model request, allowing the model to determine whether additional actions are necessary.

The distinction is fundamental: **the model determines what action to take next; the harness determines how that action is executed and whether another iteration is permitted.**

Autonomy therefore emerges not simply from tool calling, but from a self-continuing, observation-driven control loop.

## 1. The mechanics of autonomous execution

An autonomous ReAct loop follows a simple pattern:

```text
User submits objective
         |
         v
Harness initiates model request
         |
         v
Model evaluates context and task
         |
         v
   Tool call requested?
      /          \
    Yes           No
     |             |
     v             v
Validate and    Final response
execute tool       |
     |             v
     v          Stop normally
Observation
     |
     v
Update Memory and Trajectory
     |
     +----> Next model request
              (if limits permit)
```

The key is the **feedback-driven repetition**. The model sees the results of previous actions and can adapt its next decision accordingly.

The harness does not need a predefined sequence of tools. It needs a reliable mechanism for presenting state to the model, executing permitted actions, and returning their results.

In the course's implementation, `TinyAgent.run()` repeatedly invokes `_step()`. A step that executes a tool returns `None`, signaling that execution should continue. A step that receives a model response without a tool call returns the answer, terminating the loop.

Thus, a comparatively small change in orchestration creates a substantial change in behavior.

## 2. What starts, continues, and stops the loop?

Three mechanisms govern the execution lifecycle.

| Event | Decision authority | Behavior |
|---|---|---|
| **Start** | User or calling application | Invokes the agent with an objective. |
| **Continue** | Model and harness | Model requests another tool; harness executes it and initiates the next model request. |
| **Normal completion** | Model response, interpreted by harness | Model returns a response without a tool call; harness stops and reports it. |
| **Forced termination** | Harness | Execution reaches a limit, encounters a terminal failure, or is cancelled. |

A subtle but essential distinction concerns the stopping condition.

**The observation itself does not terminate execution.** Instead, the observation is returned to the model, which may request another tool or produce a final answer.

In the course's native tool-calling implementation, the absence of a tool call serves as the signal for normal completion.

This is a protocol convention, not proof that the objective has been successfully achieved. A model can produce a premature or incorrect answer. Likewise, reaching `max_steps` terminates execution without necessarily completing the task.

A robust harness should therefore distinguish **completed**, **step limit reached**, **failed**, and **cancelled** outcomes rather than treating every termination as success.

## 3. The division of responsibilities: model versus harness

The agentic harness is the software environment that makes model-directed behavior executable, observable, and bounded.

| Component | Responsibility |
|---|---|
| **LLM** | Interprets objectives, evaluates observations, selects tools and arguments, and generates responses. |
| **Harness / controller** | Orchestrates model requests, executes the loop, enforces permissions and limits, and handles stopping conditions. |
| **Tools** | Provide specific capabilities such as calculations, retrieval, database queries, and external API operations. |
| **Memory / context** | Preserves the relevant information required for subsequent model decisions. |
| **Trajectory** | Records model requests, selected actions, observations, errors, and outcomes. |
| **Planner** | Supplies planning behavior or execution configuration, such as a maximum step count. |

The model does not directly execute Python functions or operate external systems. It generates structured requests. The harness validates and executes those requests through registered tools.

This separation provides a critical architectural boundary between **decision-making authority** and **execution authority**.

The model may decide that a tool would be useful, but the harness determines whether that tool is available, whether its arguments are valid, and whether its execution is permitted.

## 4. A concrete example: investigating an operational anomaly

Consider the Parking & Ground Transportation reporting assistant receiving the request:

*"Investigate why parking occupancy is substantially below last year's level."*

An autonomous agent might proceed as follows:

1. **Reason:** Determine whether the reported decline is supported by complete, comparable data.
2. **Act:** Request a governed occupancy comparison and data-completeness check.
3. **Observe:** Occupancy appears lower, but several reporting days have missing records.
4. **Reason again:** The incomplete data may explain part of the apparent decline. Investigate data quality before pursuing an operational explanation.
5. **Act:** Request ingestion-status and reconciliation information.
6. **Observe:** Discover a failed data load affecting the comparison.
7. **Finish:** Report the evidence, explain the uncertainty, and recommend corrective review.

The important feature is that **the second investigative action depends on the first observation**.

The model is not merely following an ordered checklist defined by the developer. It is adapting the investigation based on evidence obtained during execution.

The harness enables that adaptation while restricting the agent to authorized, preferably read-only analytical tools unless additional approvals are granted.

## 5. Why autonomy requires boundaries

Autonomous continuation increases capability but also introduces risk.

An agent can misinterpret observations, repeatedly call an inappropriate tool, pursue an unproductive investigation, or propagate an initial error through subsequent actions.

The course introduces `max_steps` as a simple safety cap. A production-oriented harness should consider additional controls:

- Tool allowlisting, argument validation, and least-privilege permissions.
- Step, time, token, and cost budgets.
- Explicit cancellation and safe stopping behavior.
- Error handling and recovery policies.
- Human approval for consequential actions.
- Structured execution traces and evaluation of outcomes.

**The harness should automate continuation without surrendering control over execution.**

## 6. Connection to TinyAgent Sessions 1–4

The development progression makes the architectural transition especially clear.

- **Sessions 1–2:** The model generates responses; the application manages prompts, memory, and diagnostic trajectory.
- **Session 3:** Native tool calling adds model-selected actions and observations, but continuation remains manually controlled.
- **Session 4:** The harness automatically repeats model requests and approved tool execution until a final response or stopping boundary is reached.

The Session 3 manual Continue mechanism remains valuable as a teaching and debugging tool because it exposes every transition. Session 4 automates that mechanism rather than replacing its underlying architecture.

This also explains the course's observation that *an agent is just a for-loop*. The loop is the essential control-flow mechanism, but a useful agentic system requires the surrounding machinery that makes execution reliable and accountable.

## Takeaway

> **Agentic autonomy emerges when a harness repeatedly executes model-selected actions, returns observations, and allows the model to determine the next step without requiring manual continuation. The model supplies adaptive decision-making; the harness supplies execution, state, limits, and accountability.**

**Connection to other notes:** `the_agentic_loop_and_react.md` introduces the feedback-driven loop conceptually; `workflows_vs_agentic_systems.md` addresses the appropriate delegation of decision-making; the Session 3 retrieval note explains how model-directed tool calls first expose this distinction.

*Provenance: Based on the Session 4 course source, particularly Sections 2 and 4, and the user's documented TinyAgent Session 1–3 implementation.*