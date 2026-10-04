from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

def mcp_tool_to_llm_format(mcp_tool):
    return {
        "type": "function",
        "function":{
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.input_schema
        }

    }

async def main():
    url = "http://localhost:8081/mcp"  # Replace with your server URL

    #connect to the server using stdio
    async with streamable_http_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream,write_stream) as session:
            await session.initialize()

            #List available tools
            tools_result = await session.list_tools()
            llm_tools = [mcp_tool_to_llm_format(tool) for tool in tools_result.tools]
            print("Discovered tools:", [t["function"]["name"] for t in llm_tools])

            #let llm decide which tool to call
            # llm = ChatOllama(model = "llama3.1",temperature = 0)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=os.getenv("api_key"))

            llm_with_tools = llm.bind_tools(llm_tools)
            question = "What is 2 + 2 ?"
            system_hint = "Only use the search_knowledge_base tool if the question is specifically about AI agents, prompt engineering, or adversarial attacks on LLMs. For general knowledge, math, or unrelated questions, answer directly without using any tool."

            print(f"\nUser question: {question}")

            response = llm_with_tools.invoke(
                [
                ("system",system_hint),
                ("human",question)
                ]
            )

            #check what llm decided to do
            if not response.tool_calls:
                print("\nLLM decided no tool was needed.")
                print("Direct answer:", response.content)
                return

            tool_call = response.tool_calls[0]
            print(f"\nLLM decided to call: {tool_call['name']} with args {tool_call['args']}")

            # Step 4: execute the tool call via MCP, using the LLM's chosen arguments

            result = await session.call_tool(
                name = tool_call["name"],
                arguments = tool_call["args"]
            )
            # MCP tool results can contain text, images, audio, or resources.
            # Only text content exposes a ``text`` attribute.
            tool_output = "\n".join(
                text
                for content in result.content
                if isinstance(text := getattr(content, "text", None), str)
            )

            final_prompt = f"""the user asked: {question}
You searched and found this information:
{tool_output}

Using only the information above, write a clear, concise answer to the user's question."""            

            final_res = llm.invoke(final_prompt)
            print("\nFinal answer:", final_res.content)



if __name__ == "__main__":
    asyncio.run(main())            

