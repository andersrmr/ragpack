# From Automatic RAG to Model-Directed Retrieval: A Fundamental Transition Toward Agentic AI

**Course:** Introduction to Agents — Session 3  
**Topics:** RAG, Tool Calling, Retrieval Planning, Agent Architecture

## The central insight

Traditional RAG asks:

**"What information is semantically related to the user's question?"**

Model-directed retrieval allows the LLM to ask:

**"What information do I need to accomplish the user's objective, and what action should I take to obtain it?"**

This distinction represents a fundamental architectural transition from passive retrieval to model-directed decision-making.

## 1. The limitation of basic automatic RAG

In the course's initial implementation, the harness automatically retrieves documents based on the latest user prompt and prepends them to the LLM's context.

This introduces several limitations:

- Retrieval happens whether or not additional information is needed.
- The retrieval query is based on the user's latest words, which may not fully represent their intent.
- Semantically relevant documents may not contain the information necessary to answer the question.
- Retrieval does not independently determine what evidence, calculations, or operations are required to accomplish the user's objective.

The course illustrates this with a pronoun-resolution example: a follow-up question using "she" retrieves facts about the wrong person.

However, this particular example is somewhat contrived. A context-aware retrieval pipeline could resolve the pronoun using conversation history or rewrite the query before retrieval. This limitation is not intrinsic to RAG.

The deeper issue is the **fixed retrieval policy**, not simply a failure of language understanding.

## 2. A more instructive example: airport operations

Consider an assistant asked:

"Were yesterday's flight cancellations unusually high, and did they affect passenger volumes?"

**Basic automatic RAG** might retrieve documents about flight cancellations, airport operations, and passenger traffic. These could be semantically relevant without providing the quantitative evidence necessary to answer the question.

**Model-directed tool use** allows the LLM to recognize that the request requires several operations:

1. Retrieve yesterday's flight cancellation count.
2. Establish a suitable historical baseline.
3. Retrieve observed passenger volumes and comparable baselines.
4. Compare the results, investigate temporal relationships, and identify anomalies.
5. Synthesize an answer while distinguishing observed associations from causal conclusions.

The crucial difference is that the LLM identifies the information and operations needed to answer the question, rather than simply receiving documents judged relevant to its wording.

This example also illustrates the progression beyond document search toward structured data retrieval, calculations, and analytical tools.

## 3. Retrieval versus retrieval planning

| Dimension | Basic automatic RAG | Model-directed retrieval |
|---|---|---|
| Retrieval initiation | Harness | LLM |
| Query formulation | Typically user text | Model-generated query |
| Retrieval timing | Every message | When the model chooses |
| Information selection | Fixed retrieval procedure | Based on interpreted task requirements |
| Multiple operations | Requires predefined orchestration | Model can request successive tools |
| Principal advantage | Simplicity and predictability | Flexibility and task awareness |
| Principal risk | Irrelevant or insufficient retrieval | Poor tool selection or omitted retrieval |

Importantly, sophisticated RAG systems can also perform query rewriting, intent classification, routing, and multistep retrieval.

**RAG and tool calling are not competing technologies.** Retrieval can itself be exposed as a tool. What changes is who controls the retrieval decision.

## 4. The architectural significance

This distinction clarifies the separation of responsibilities among three components:

**The LLM** interprets the user's intent, determines what information or action is required, and generates a structured tool request.

**The harness** supplies available tool schemas, validates requests, executes permitted operations, and returns observations.

**The tools** provide capabilities such as document search, database queries, calculations, file operations, or external service access.

The model does not execute these operations directly. It requests them; the harness performs them.

This separation is fundamental to understanding agentic AI.

## 5. Important qualifications

Model-directed retrieval is not inherently superior to automatic RAG.

It introduces additional model calls, latency, cost, and opportunities for error. The model may choose an inappropriate tool, generate a poor retrieval query, or fail to request information when necessary.

Automatic retrieval remains useful when the information requirements are predictable and reliable retrieval should occur for every request.

Hybrid architectures can combine deterministic retrieval policies with model-directed tool use.

## 6. Connection to TinyAgent Session 3

Session 3 introduces the mechanisms necessary for this transition:

- `NativeTools` exposes registered Python functions through schemas.
- The LLM returns structured function calls.
- The harness executes the requested function.
- The observation is returned to conversation memory.
- `Trajectory` records the action and result.
- `Continue` allows the model to take another step.

At this stage, TinyAgent executes **one action per step**. The user explicitly controls continuation.

Session 4 introduces the automated loop that allows successive model-directed steps without requiring manual continuation.

Therefore, Session 3 establishes the essential mechanics of agentic behavior without yet implementing autonomous execution.

## 7. Broader lesson

The conceptual progression is:

**Semantic relevance → Information requirements → Action selection → Observation → Informed response**

The progression is not merely about giving an LLM access to more information. It is about enabling the LLM to participate in deciding how information should be acquired and used.

That is a foundational distinction between conventional retrieval-augmented applications and agentic systems.

### Key takeaway

**Agentic AI is not defined simply by its ability to call tools. Its deeper significance lies in delegating selected decisions about information acquisition and action to the model, while retaining execution, validation, and control within the harness.**