from lib.coding_agent import coding_agent as _coding_agent, log
from openai import OpenAI
from lib.utils import create_sandbox
from lib.tools import execute_code
from lib.tools_schemas import execute_code_schema
from lib.logger import logger
from lib.ui import ui
import threading


def coding_agent_demo_cli():
    client = OpenAI()
    sbx = create_sandbox()
    messages = []
    logger.info("✨: Hello there! Ask me to code something!")
    while (query := input(">:")) != "/exit":
        messages, usage = log(
            _coding_agent,
            query=query,
            messages=messages,
            client=client,
            tools_schemas=[execute_code_schema],
            system="You are senior Python software engineer",
            tools={"execute_code": execute_code},
            sbx=sbx,
        )


def coding_agent_demo_ui():
    client = OpenAI()
    sbx = create_sandbox()
    messages = []
    demo = ui(
        _coding_agent,
        messages=[],
        client=client,
        tools_schemas=[execute_code_schema],
        system="You are senior Python software engineer",
        tools={"execute_code": execute_code},
        sbx=sbx,
    )
    demo.launch(height=800, share=True)
