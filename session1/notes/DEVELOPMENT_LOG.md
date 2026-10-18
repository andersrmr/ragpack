# TinyAgent Development Log

## Session 1 — OpenAI API Integration

**Course:** Introduction to Agents  
**Project:** TinyAgent Tutor  
**Coding assistant:** OpenAI Codex

### Objective
Adapt the course's TinyAgent implementation to use the OpenAI API instead of Ollama.

### Changes implemented
- Created `tinyagent.py` using the course architecture.
- Integrated the official OpenAI Python SDK.
- Configured API access using `OPENAI_API_KEY`.
- Made the model configurable using `TINYAGENT_MODEL`.
- Added tests and setup documentation.

### Validation
- Successfully connected to an OpenAI model.
- Confirmed response text and token metadata.
- Ran offline tests successfully.

### Key architectural lesson
The LLM is responsible for generating responses, while the TinyAgent harness manages conversation history and orchestrates the interaction.

### Next steps
- Implement a minimal terminal chat interface.
- Add `/clear` and `/quit` commands.
- Introduce a course-specific tutoring system prompt.