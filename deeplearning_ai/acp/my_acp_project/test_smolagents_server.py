# Wrapping a Smolagents Agent into an ACP Server

from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool
import logging 
from dotenv import load_dotenv

load_dotenv() 

server = Server() # Wrapping a Smolagents Agent into an ACP Server

model = LiteLLMModel(
    model_id="openai/gpt-4",  
    max_tokens=2048
)

@server.agent()
async def genai_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is a CodeAgent which supports the department to handle multimodal LLM questions for analysts.  Current or prospective analysts can use it to find answers about their research issues with GenAI." 
    agent = CodeAgent(tools=[DuckDuckGoSearchTool(), VisitWebpageTool()], model=model)

    prompt = input[0].parts[0].content
    response = agent.run(prompt)

    yield Message(parts=[MessagePart(content=str(response))])


if __name__ == "__main__":
    server.run(port=8000)
