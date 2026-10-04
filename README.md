# Adaptive RAG MCP Client-Server System

An implementation of the **Model Context Protocol (MCP)** using LangChain, ChromaDB, and Google Gemini. This project decouples local knowledge retrieval tools into an independent MCP server (`server.py`) and exposes them to an intelligent client (`client.py`) capable of dynamic tool-selection and routing.

---

## Project Structure

```text
Mcp/
├── connecting mcp to llm/
│   ├── client.py        # MCP Client with Gemini binding & evaluation logic
│   └── server.py        # MCP Server exposing vector store retrieval tools
├── chroma_db/           # Local vector database storage (Ignored by Git)
├── .env                 # Environment variables containing API keys (Ignored by Git)
└── README.md
