# ltc_acp_server.py
# acp_sdk needs python 3.11 and above

import os
import re
from collections.abc import AsyncGenerator
from pathlib import Path

import nest_asyncio
import acp_sdk
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
from crewai import Agent, Crew, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pypdf import PdfReader

nest_asyncio.apply()

# Load environment variables from ~/.env
env_path = Path("~/.env").expanduser()
load_dotenv(dotenv_path=env_path)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in ~/.env file")

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini")

# --- RAG Setup from ltc_rag_agent_openai.ipynb ---

def get_pdf_filenames_from_group(group_file_path):
    """Extracts PDF filenames from a given group file."""
    try:
        with open(group_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Regex to find filenames like 'Processing paper 1 - filename.pdf, 1234 characters'
        filenames = re.findall(r"Processing paper \d+ - ([^,]+?\.pdf), \d+ characters", content)
        print(f"Identified {len(filenames)} PDF filenames.")
        return filenames
    except FileNotFoundError:
        print(f"Error: Group file not found at {group_file_path}")
        return []

def read_pdf_content(file_path):
    """Reads and extracts text content from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        print(f"Read {len(text)} characters from {file_path}")
        return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Define paths and load documents
GROUP_FILE = "/Users/ytchen/Documents/projects/research/NYCU/ltc/classified_articles/Long-Term Care Policies and Programs.txt"
PDF_DIR = "/Users/ytchen/Documents/projects/research/NYCU/ltc/data/pdfs/"

pdf_filenames = get_pdf_filenames_from_group(GROUP_FILE)
documents = []
for filename in pdf_filenames:
    file_path = os.path.join(PDF_DIR, filename.strip())
    if os.path.exists(file_path):
        content = read_pdf_content(file_path)
        if content and content.strip():
            documents.append(content)

if not documents:
    raise ValueError("No documents were loaded. Please check the group file and PDF paths.")

# Split documents and create vector store
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.create_documents(documents)
print(f"Created {len(splits)} text splits.")

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()

# --- CrewAI Tool, Agent, and Server Setup ---

class LTC_RAG_Tool(BaseTool):
    name: str = "Long-Term Care Policy RAG Tool"
    description: str = "Searches and returns relevant information about long-term care policies. Use it to answer questions about policies, regulations, and procedures."

    def _run(self, question: str) -> str:
        docs = retriever.invoke(question)
        return "\n\n".join(doc.page_content for doc in docs)

rag_tool = LTC_RAG_Tool()

ltc_policy_expert = Agent(
    role="Long-Term Care Policy Expert",
    goal="Provide accurate information about long-term care policies based on the provided documents.",
    backstory="An expert in Taiwanese long-term care policies, skilled at interpreting official documents and answering questions clearly.",
    tools=[rag_tool],
    llm=llm,
    verbose=True,
)

server = Server()

@server.agent()
async def ltc_rag_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    An agent for answering questions about Taiwanese long-term care policies.
    It uses a RAG pipeline to find answers from a curated set of policy documents.
    """
    user_question = input[0].parts[0].content

    task = Task(
        description=f"Answer the following question: {user_question}. Use the provided context to formulate a clear and concise answer.",
        expected_output="A comprehensive and accurate response based on the provided long-term care policy documents.",
        agent=ltc_policy_expert,
    )

    crew = Crew(agents=[ltc_policy_expert], tasks=[task], verbose=True)
    
    task_output = await crew.kickoff_async()
    
    yield Message(parts=[MessagePart(content=str(task_output))])

if __name__ == "__main__":
    print("Starting the LTC RAG ACP server...")
    server.run(port=8002)
