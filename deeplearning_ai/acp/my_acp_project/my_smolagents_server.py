from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool, ToolCallingAgent, ToolCollection
from mcp import StdioServerParameters

server = Server()

model = LiteLLMModel(
    model_id="openai/gpt-4",  
    max_tokens=2048
)

server_parameters = StdioServerParameters(
    command="uv",
    args=["run", "test_mcpserver.py"],
    env=None,
)

@server.agent()
async def genai_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is a CodeAgent which supports the department to handle multimodal LLM questions for analysts.  Current or prospective analysts can use it to find answers about their research issues with GenAI." 
    agent = CodeAgent(tools=[DuckDuckGoSearchTool(), VisitWebpageTool()], model=model)

    prompt = input[0].parts[0].content
    response = agent.run(prompt)

    yield Message(parts=[MessagePart(content=str(response))])

@server.agent()
async def vet_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is a Vet Agent which helps users find vets near them."
    with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
        agent = ToolCallingAgent(tools=[*tool_collection.tools], model=model)
        prompt = input[0].parts[0].content
        response = agent.run(prompt)

    yield Message(parts=[MessagePart(content=str(response))])

if __name__ == "__main__":
    server.run(port=8000)
