from colorama import Fore
from mcp.server.fastmcp import FastMCP
import json 
import requests

mcp = FastMCP("vetserver")

# Build server function
@mcp.tool()
def list_vets(state:str) -> str:
    """This tool returns vets that may be near you.
    Args:
        state: the two letter state code that you live in. 
        Example payload: "CA"

    Returns:
        str: a list of vets that may be near you
        Example Response "{"VET001":{"name":"Dr. Emily Smith", "specialty":"Small Animal Internal Medicine"...}...}" 
        """

    url = 'https://raw.githubusercontent.com/yu-to-chen/generative-ai/refs/heads/main/deeplearning_ai/acp/data/veterinarians.json'
    resp = requests.get(url)
    vets = json.loads(resp.text)

    matches = [vet for vet in vets.values() if vet['address']['state'] == state]    
    return str(matches) 

# Kick off server if file is run 
if __name__ == "__main__":
    mcp.run(transport="stdio")
