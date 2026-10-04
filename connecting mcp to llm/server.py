from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field
from typing import Annotated
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

mcp = MCPServer(name = "Adaptive-rag-tools")

embedding_model = OllamaEmbeddings(model = "nomic-embed-text")
vectorstore = Chroma(
    collection_name="adaptive-rag",
    embedding_function=embedding_model,
    persist_directory="./chroma_db",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

@mcp.tool(title="Search the knowledge base")

def search_knowledge_base(query: Annotated[str, Field(description="The question or topic to search for.")]) -> str:
    """Search the local vector store for documents relevant to a question about AI agents, prompt engineering, or adversarial attacks on LLMs."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )
if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8081, host="0.0.0.0")   
