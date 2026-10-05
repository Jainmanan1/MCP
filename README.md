# MCP Server/Client: Adaptive Tool Selection

A minimal implementation of the Model Context Protocol (MCP), built on the
official MCP Python SDK v2, demonstrating a RAG retriever exposed as a
standalone tool and called dynamically by an LLM via tool-binding — rather
than hardcoded in application logic.

## Overview

This project decouples a local knowledge-retrieval capability into an
independent MCP server, then connects it to an LLM client that decides,
on its own, whether and when to use the tool — the core idea behind MCP:
letting a model discover and use tools dynamically, rather than a developer
hand-wiring "if question is about X, call Y."

## Project Structure
Mcp/
├── connecting mcp to llm/
│   ├── client.py        # MCP Client with Gemini binding & evaluation logic
│   └── server.py        # MCP Server exposing vector store retrieval tools
├── chroma_db/           # Local vector database storage (Ignored by Git)
├── .env                 # Environment variables containing API keys (Ignored by Git)
└── README.md


## How it works

1. `server.py` exposes a `search_knowledge_base` tool over HTTP (`streamable-http`
   transport) backed by a Chroma vector store (reused from the
   [adaptive-rag-langgraph] project).
2. `client.py` connects to the server, discovers available tools via
   `session.list_tools()`, and binds them to an LLM with `bind_tools`.
3. The LLM — not the client code — decides whether the user's question
   requires the tool, and if so, generates the search query itself.
4. If a tool call is made, the client executes it via the MCP session and
   feeds the result back to the LLM for a final, synthesized answer.

## Key Finding: Tool-calling judgment varies significantly by model

Tested the same MCP tool, bound identically via `bind_tools`, against two
different LLMs with a question requiring no retrieval at all ("What is 2 + 2?"):

- **llama3.1 8B (local, via Ollama):** called the tool anyway — even when
  explicitly instructed in a system message not to use it for unrelated
  questions. The tool call persisted across both a passive hint and a direct
  instruction, indicating a model-level bias toward tool use rather than a
  prompt-wording issue.
- **Gemini 2.5 Flash (Google AI Studio, free tier):** correctly recognized no
  tool was needed and answered directly ("2 + 2 is 4"), with no special
  instruction required.

**Conclusion:** MCP's discovery and invocation mechanism worked identically
and correctly in both cases — the difference was entirely in model judgment
about *when* to use an available tool. This validates an architectural choice
made in the companion [adaptive-rag-langgraph] project: explicit,
constrained routing (`Literal["vectorstore", "web_search", "direct_answer"]`
via structured output) is a more reliable pattern than open-ended `bind_tools`
judgment when working with smaller local models — because it narrows the
model's decision to a fixed classification rather than free tool selection.

## Setup

**Prerequisites**
- [Ollama](https://ollama.com) running locally with `nomic-embed-text` pulled
  (for the vector store's embeddings)
- A Google AI Studio API key ([aistudio.google.com](https://aistudio.google.com)) — free tier, no credit card required



Create a `.env` file:


## Known Limitations

- `bind_tools` tool-selection judgment is model-dependent; see the finding
  above. This project does not implement the explicit-routing mitigation —
  see adaptive-rag-langgraph for that pattern.
- The reused vector store carries a known chunking issue (some retrieved
  content is page boilerplate rather than article text) — documented in the
  adaptive-rag-langgraph repo.
