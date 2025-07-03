# Wrapping the RAG Agent into an ACP Server

from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server

from crewai import Crew, Task, Agent, LLM
from crewai_tools import RagTool

import nest_asyncio
nest_asyncio.apply()

server = Server()
llm = LLM(model="ollama_chat/qwen2.5:14b", base_url="http://localhost:11434", max_tokens=8192)

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "qwen2.5:14b",
        }
    },
    "embedding_model": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    }
}
rag_tool = RagTool(config=config,  
                   chunk_size=1200,       
                   chunk_overlap=200,     
                  )
rag_tool.add("../data/王梓蓉_論文改稿_v2.pdf", data_type="pdf_file")


@server.agent()
async def research_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is an agent for questions around a research thesis, it uses a RAG pattern to find answers based on the thesis. Use it to help answer questions on the thesis and its research."

    thesis_agent = Agent(
        role="Senior thesis qa Assistant", 
        goal="Understand the content of a thesis",
        backstory="You are an expert qa agent designed to assist with research queries for a particular thesis",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[rag_tool], 
        max_retry_limit=5
    )

    task1 = Task(
         description=input[0].parts[0].content,
         expected_output = "A comprehensive response as to the users question",
         agent=thesis_agent
    )
    crew = Crew(agents=[thesis_agent], tasks=[task1], verbose=True)
    
    task_output = await crew.kickoff_async()
    yield Message(parts=[MessagePart(content=str(task_output))])

if __name__ == "__main__":
    server.run(port=8001)
