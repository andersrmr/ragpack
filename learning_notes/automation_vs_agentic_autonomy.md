# Automation vs. Agentic Autonomy: Who Controls the Execution Path?

**Course:** Introduction to Agents — Sessions 1 and 4  
**Themes:** Automated workflows, model-directed control flow, adaptive decision-making, delegated autonomy, operational risk

## The central insight

**The fundamental distinction between conventional automation and agentic autonomy is not whether a system operates without human intervention, but who determines its sequence of actions.**

In a conventional automated workflow, the developer specifies the execution logic in advance, including the sequence of operations, conditions, branches, and recovery procedures.

In an agentic system, some decisions about the next action are delegated to the model. Based on the user's objective and observations from previous actions, the model dynamically selects among available tools and determines how to proceed.

Both systems may operate automatically. The difference is **automation of execution versus autonomy in selecting actions**.

This is a distinction in control architecture, not a claim that agentic systems are inherently more sophisticated or effective.

## 1. Conventional automation can be highly sophisticated

Consider a scheduled data pipeline that retrieves operational data from an external API.

The pipeline might:

1. Execute every 30 minutes.
2. Authenticate to the source API.
3. Request data and follow pagination cursors.
4. Retry failed requests using exponential backoff.
5. Persist checkpoints to support recovery.
6. Validate the response and write records to cloud storage.
7. Record operational metrics and report failures.

This system may contain substantial conditional logic, state management, and error recovery.

It can operate continuously, recover from interruptions, and make decisions such as whether to retry, stop, or resume from a saved cursor.

Nevertheless, it remains conventional automation if those decisions are governed by predefined program logic.

**Complexity, branching, and unattended execution do not, by themselves, constitute agentic autonomy.**

The Samsara GPS ingestion work provides a practical example: an ADF pipeline or Databricks Python collector can implement sophisticated pagination, checkpointing, and recovery without requiring an LLM to determine its next action.

For predictable ingestion requirements, that is generally a strength rather than a limitation.

## 2. What changes when action selection is delegated to a model?

Now consider a different objective:

*"Investigate why the Samsara GPS ingestion process has stopped producing fresh data."*

A conventional monitoring workflow might check predefined conditions—job status, error codes, elapsed time since the last successful load—and execute corresponding alerts or retries.

An agentic investigation could have greater flexibility:

1. Retrieve recent job execution logs.
2. Observe that the collector completed successfully but produced unexpectedly few records.
3. Check source API response metadata and cursor progression.
4. Discover that the cursor advanced but the returned data window was unusually small.
5. Compare recent collection volumes against historical patterns.
6. Determine which additional diagnostic check would help distinguish a source-data issue from an ingestion defect.
7. Report findings and recommend further action.

The agent's sequence is not entirely prescribed in advance. Later investigations depend on the evidence discovered in earlier steps.

The model is delegated authority to select diagnostic actions, while the harness restricts it to approved tools and operations.

This is an example of **bounded autonomy**: flexible analytical decision-making within controlled execution boundaries.

## 3. Parking & GT: the progression from workflow to agent

The Parking & Ground Transportation Revenue Reporting Copilot provides another concrete illustration.

Its core responsibilities include collecting operational data, calculating parking occupancy and trip metrics, performing year-over-year comparisons, and generating presentation materials.

These activities naturally support a deterministic workflow.

| System | Who determines the next action? | Parking & GT example |
|---|---|---|
| **Automated workflow** | Predefined code and business rules | Calculate approved metrics, generate charts, and populate recurring PowerPoint reports. |
| **AI-powered assistant** | Primarily the user, supported by model interpretation | Answer a question about a reporting variance or draft an executive narrative. |
| **Agentic system** | Model selects among permitted analytical actions | Investigate an unexpected occupancy decline by deciding which data-quality, historical, or operational checks to perform. |

The progression is not simply from a less capable system to a more capable one.

Each architecture addresses different requirements.

The reporting process contains many operations that should remain deterministic: metric definitions, reporting periods, approved formulas, data reconciliation, and prescribed presentation outputs.

Potential value from agency arises where the appropriate investigative path cannot be fully anticipated.

For instance, an unexpected occupancy decline might reflect incomplete source data, calendar effects, parking capacity changes, or genuine demand shifts.

A bounded agent could select diagnostic tools based on what each successive result reveals, instead of following the same investigation sequence regardless of the evidence.

## 4. Automation and agency are complementary, not competing architectures

An agentic system is itself implemented using automation.

The harness automatically invokes the model, validates tool requests, executes approved functions, records observations, and continues the interaction.

What distinguishes the system is that **selected decisions governing the execution path are supplied dynamically by the model rather than fully prescribed in code**.

This distinction also allows hybrid architectures.

For example:

- A deterministic data pipeline ingests and validates records.
- A scheduled quality-control process detects an unexpected anomaly.
- A bounded agent investigates the anomaly using approved read-only tools.
- A human reviews the findings and authorizes consequential corrective action.

Here, conventional automation provides predictable and testable operations, while the agent supplies flexibility in an uncertain analytical task.

The architecture delegates model judgment only where that judgment is useful.

## 5. Increased autonomy introduces additional risk

Model-directed action selection creates opportunities for adaptive problem-solving, but it also introduces uncertainty into the execution path.

A model may select an inappropriate tool, misinterpret a result, repeat an ineffective action, overlook an important check, or reach an unsupported conclusion.

Risk becomes more significant when the agent can modify production data, launch jobs, change configurations, or communicate externally.

A sound design therefore distinguishes between:

- **Decision authority:** What may the model decide?
- **Execution authority:** What operations may the harness actually perform?
- **Approval authority:** Which actions require human confirmation?

The harness should enforce permissions, validate requests, constrain resource usage, and record observable execution traces.

Human oversight is particularly important at consequential boundaries, rather than merely reviewing the final output of an otherwise unrestricted process.

## 6. The architectural decision: when is autonomy warranted?

The central design question is:

**Does the task require the model to determine its next operation based on information discovered during execution?**

If the sequence is predictable, ordinary automation is often more reliable, economical, and auditable.

If the task involves ambiguity, changing evidence, or investigative decisions that are difficult to encode as fixed rules, bounded agentic autonomy may provide meaningful advantages.

This is not an absolute distinction. Conventional workflows can support sophisticated rules and adaptive branching, while agents can contain deterministic subroutines. The practical question is where model-directed decision-making adds sufficient value to justify its additional complexity and risk.

## 7. Connection to the TinyAgent learning progression

Session 1 introduced the distinction between workflows and agents conceptually.

Session 3 demonstrated how the model can select a tool and receive an observation, but still relied on the user to initiate the next step.

Session 4 implements the self-continuing loop, allowing model-directed action selection to govern multiple successive operations.

The progression clarifies that **tool use, autonomous execution, and business-process automation are separate capabilities that can be combined in different ways**.

A system does not become meaningfully agentic simply because it includes an LLM or runs automatically.

## Takeaway

> **Automation executes a process whose decision logic is predefined; agentic autonomy delegates selected decisions about what to do next to a model. Both can run without human intervention. The objective is not maximum autonomy, but appropriate autonomy: preserve deterministic controls where the work is predictable, and introduce bounded model-directed decision-making where adaptability creates real value.**

**Connection to other notes:** `workflows_vs_agentic_systems.md` introduces this distinction and its risk implications; `the_agentic_loop_and_react.md` explains how observation-driven action selection works; the Session 4 harness note explains how that behavior becomes self-continuing.

*Provenance: Synthesized from the Session 1 workflow-versus-agent discussion, the Session 4 autonomy material, and the previously discussed ATL Parking & GT and Samsara examples. The operational examples are illustrative, not claims about deployed agentic functionality.*