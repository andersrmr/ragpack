# The Agentic Loop and ReAct: The Defining Architectural Idea

**Learning notes | Introduction to Agents — Lecture 1, slides 64–78**  
**Themes:** ReAct, model-directed control flow, tools, observations, memory, and stopping

## The central insight

**The agentic loop is the defining architectural idea.** An agent does not merely produce one answer or execute a predetermined sequence of steps. It can repeatedly **assess the current state, choose an action, observe what happened, and use that feedback to decide the next step**.

This feedback-driven *control flow* is more important to the definition of agency than whether the system has a chat interface, uses an LLM, calls a tool once, or can access many tools.

## The ReAct pattern

**ReAct = reasoning + acting.** In a simple conceptual form:

```text
User objective + available context
              |
              v
     Reason: What is needed next?
              |
              v
     Act: Request a permitted tool
              |
              v
     Observe: Inspect the tool result
              |
              +----> Update context and repeat as needed
              |
              v
     Final answer / completion / escalation
```

An iteration can be described as **Thought → Action → Observation**. In practical implementations, the “thought” is the model's decision or action proposal; we do **not** need access to hidden chain-of-thought text to inspect whether it chose a sensible tool. The harness records the visible action request, its inputs, and the result.

The agentic part is the *feedback*: an observation can change the next action. A fixed workflow can call exactly the same tools, but the order and branches would already have been specified by its developer.

## A concrete example: an unexpected reporting anomaly

Suppose the Parking & GT assistant is asked: “Why is this month's parking occupancy materially below last year's?”

1. **Reason:** The model decides that it first needs to establish whether the reported decline is real.
2. **Act:** It requests a governed occupancy query and a source-completeness check.
3. **Observe:** The query shows a decline, but the completeness check reports missing records for several days.
4. **Reason again:** The model changes course: first investigate ingestion and data quality, rather than offering an operational explanation.
5. **Act and observe:** It requests the relevant job-status or reconciliation information.
6. **Finish or escalate:** It reports the evidence, identifies remaining uncertainty, and recommends review instead of asserting a misleading business conclusion.

The key feature is not the specific query. It is that **later actions depend on what earlier actions revealed**. Real system implementations must honor configured permissions and obtain approvals for consequential changes or distribution.

## What makes the loop possible?

| Component | Role in the architecture |
|---|---|
| **LLM / decision policy** | Interprets the objective and chooses an available next action or final answer. |
| **Tools** | Give the agent permitted ways to inspect or change its environment (query data, search documents, calculate, etc.). |
| **Harness / controller** | Validates requests, executes tools, adds observations to context, manages the loop and enforces limits. |
| **Memory / context** | Carries relevant prior messages, results, and state into subsequent decisions. |
| **Trajectory / trace** | Records what was requested, what executed, what returned, and what the agent concluded. |
| **Stop conditions / guardrails** | End successful work, cap steps and costs, handle failures, and route sensitive actions for approval. |

**Tools, planning, and memory support an agent; none, by itself, makes a system agentic.** A one-time tool call is not the same as an autonomous, observation-driven loop.

## Why the loop introduces both capability and risk

Repeated observation and decision-making enables adaptive investigation, but also allows errors to propagate across steps. An agent can select the wrong tool, misunderstand an observation, retry unnecessarily, or pursue an unproductive sequence. More iterations do not automatically mean a better result.

Good harness design includes registered/allowlisted tools, argument validation, explicit permissions, finite step budgets, timeouts, error recovery, observable traces, and a way to stop or escalate. When actions affect data, resources, communications, or stakeholders, require the appropriate approval boundary.

## Connection to our TinyAgent implementation

- **Sessions 1–2:** TinyAgent produces responses, captures conversation context, and records diagnostic trajectory; quiz behavior is manually orchestrated.
- **Session 3:** Native tool calling adds structured actions and observations. The harness can execute **one action per step**, then **stops**; the user presses **Continue** to request the next model step. This exposes the mechanics while keeping control explicit.
- **Session 4:** An automated controller can repeatedly request model decisions and execute approved tool steps until a final response or stopping condition. This is where the agentic loop becomes self-continuing.

A key teaching advantage of Session 3 is that every transition is inspectable: **model request → tool call → harness execution → observation → Continue → next model decision**. The next step in learning is to automate that sequence without losing observability or control.

## Takeaway

> **An agent is distinguished not just by having tools, but by using observations from one action to determine the next action within a controlled feedback loop. The harness—not the model alone—makes that loop executable, bounded, and accountable.**

**Connection to other notes:** `workflows_vs_agentic_systems.md` addresses when delegated control is appropriate; `rag_vs_model_directed_retrieval.md` illustrates the same principle in retrieval and information acquisition.

*Provenance: Synthesized from our earlier discussion of Lecture 1 slides 64–78 and the TinyAgent Session 1–3 progression. The original slide deck was not re-opened for this note.*
