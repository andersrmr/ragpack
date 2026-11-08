# From Automated Workflows to Agentic Systems

**Learning notes | Introduction to Agents — Lecture 1, slide 62**  
**Themes:** Autonomy, control flow, human oversight, risk, and appropriate system design

## The central distinction

**An agentic system differs from a conventional automated workflow primarily in who decides what to do next.** In a workflow, the developer defines the sequence and decision rules. In an agentic system, a model is delegated some authority to choose actions based on the task and observations, within boundaries established by the surrounding software and organization.

This is a spectrum of *delegated decision-making*, not a hierarchy in which the most autonomous design is always best.

## Three useful stages

| Stage | Who controls the next step? | Typical capabilities | Example in Parking & Ground Transportation (GT) reporting |
|---|---|---|---|
| **Automated workflow** | Predefined code, rules, and schedule | Ingest data, calculate metrics, create charts, populate reports, run quality checks | A repeatable process calculates monthly parking occupancy and GT trip activity, compares year-over-year metrics, and prepares PowerPoint reports. |
| **AI-powered assistant** | Primarily the user, with model assistance | Answer questions, interpret trends, draft narratives, explain metrics, assist with exceptions | A staff member asks why occupancy changed; the assistant interprets approved metrics and drafts an executive summary for review. |
| **Agentic system** | The model chooses among authorized next actions, subject to guardrails | Select tools, investigate discrepancies, adapt analysis after results, request approval when needed | Given an unusual reporting variance, the model decides which governed datasets and checks to consult, compares possible explanations, and recommends an action. |

These categories overlap. A workflow can contain an LLM step without becoming an agent. An agent can also operate *inside* a larger deterministic workflow. A scheduled, end-to-end report generator may remain conventional automation if all branches and actions are predefined.

## Parking & GT: why autonomy should be introduced deliberately

The Revenue Reporting Copilot has a clear, repeatable core: operational inputs → metric calculations → year-over-year comparison → quality assurance → executive narrative and PowerPoint production. Much of this should remain **deterministic and testable**: formulas, reporting periods, source reconciliation, and prescribed report outputs.

The potential value of agency begins where the next analytical step is not known in advance. For example, an unexpected decline in parking occupancy might warrant checking source completeness, day-of-week mix, special events, and related operational data. A bounded agent could choose which diagnostic check to run next *based on the previous result*. It should not silently redefine metrics, alter production data, or distribute a consequential report without the appropriate authorization.

A practical evolution discussed for this project is: personal GPT/prototype → shared Databricks App → governed operational data and evaluated assistant → narrowly approved agentic workflow. **The use case and control requirements should drive the architecture and platform choice, not the other way around.** This is an illustrative progression, not a requirement that every stage must be implemented.

## The autonomy–risk tradeoff

As delegated autonomy increases, the system can handle more ambiguous tasks, but it can also make more consequential mistakes. Risk comes from both *bad reasoning* (selecting the wrong next action) and *bad execution* (performing an action with excessive scope or insufficient verification).

A sound design makes these questions explicit:

- **Authority:** Which decisions may the model make, and which require approval?
- **Tools and permissions:** What can it read, calculate, write, or distribute? Apply least privilege.
- **Evaluation:** How do we measure correctness, usefulness, error rates, and recovery behavior?
- **Observability:** Can we see the requested action, arguments, tool result, and final output?
- **Stop and recovery:** What happens if data are missing, a tool fails, or the agent loops without progress?
- **Human accountability:** Who reviews metrics, resolves exceptions, and approves outward-facing or consequential actions?

Risk is not eliminated by inserting a human at the end of an opaque process; it is reduced by setting appropriate boundaries **throughout** the process.

## A useful design test

Ask: **Does the task require the model to choose its next operation based on what it discovers?**

If not, ordinary automation—possibly with an LLM writing or explaining results—may be more economical, predictable, and auditable. If yes, consider granting *bounded* agentic control over the uncertain portion, while keeping stable business rules in deterministic code.

## Takeaway

> **The objective is not maximum autonomy. It is appropriate autonomy: delegate the decisions that benefit from model judgment, and retain deterministic controls, permissions, evaluation, and human oversight where they matter.**

**Connection to other notes:** `the_agentic_loop_and_react.md` explains how model-directed action selection operates; `rag_vs_model_directed_retrieval.md` explains how this shift changes information retrieval.

*Provenance: Synthesized from our earlier discussion of Lecture 1 slide 62 and the ATL Parking & GT reporting example. The original slide deck was not re-opened for this note.*
