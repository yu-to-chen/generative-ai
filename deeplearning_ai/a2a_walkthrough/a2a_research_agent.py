import os

import uvicorn
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import google_search

from helpers import setup_env

setup_env()

PORT = int(os.getenv("RESEARCH_AGENT_PORT"))
HOST = os.getenv("AGENT_HOST")

"""
def main() -> None:
    # Create the Agent
    root_agent = LlmAgent(
        model="gemini-3-pro-preview",
        name="HealthResearchAgent",
        tools=[google_search],
        description="Provides healthcare information about symptoms, health conditions, treatments, and procedures using up-to-date web resources.",
        instruction="You are a healthcare research agent tasked with providing information about health conditions. Use the google_search tool to find information on the web about options, symptoms, treatments, and procedures. Cite your sources in your responses. Output all of the information you find.",
    )

    # Make the agent A2A-compatible
    a2a_app = to_a2a(root_agent, host=HOST, port=PORT)
    print("Running Health Research Agent")
    uvicorn.run(a2a_app, host=HOST, port=PORT)
"""

def main() -> None:
    # 1. Ensure the Project ID is in the environment for Vertex AI
    import os
    os.environ["GOOGLE_CLOUD_PROJECT"] = "seismic-bucksaw-396319"
    os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1" # Or your specific region

    # 2. Add the 'vertex_ai/' prefix to the model name
    root_agent = LlmAgent(
        #model="vertex_ai/gemini-1.5-pro", # Use vertex_ai/ prefix
        #model="vertex_ai/gemini-2.5-flash",
        #model="vertex_ai/gemini-2.0-flash-lite-001", 
        #model="vertex_ai/gemini-1.5-flash",
        model="google/gemini-2.0-flash-lite-001",
        name="HealthResearchAgent",
        tools=[google_search],
        description="Provides healthcare information...",
        instruction="...",
    )

    # Make the agent A2A-compatible
    a2a_app = to_a2a(root_agent, host=HOST, port=PORT)
    print("Running Health Research Agent")
    uvicorn.run(a2a_app, host=HOST, port=PORT)

if __name__ == "__main__":
    main()
