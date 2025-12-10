# E2B Code Interpreter + OpenAI Documentation

Quick reference for building coding agents with E2B sandboxes and OpenAI. Attach this file to Jupyter AI prompts for accurate code generation.

---

## Installation

```bash
pip install e2b-code-interpreter==2.2.0
pip install openai==2.4.0
pip install python-dotenv
```

---

## Configuration

**Environment Variables:**
- `OPENAI_API_KEY`: OpenAI API key for LLM access
- `E2B_API_KEY`: E2B API key for sandbox creation

**Initialization:**
```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from e2b_code_interpreter import Sandbox

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

---

## Core Classes

### OpenAI Client

**Initialization:**
```python
from openai import OpenAI

client = OpenAI()
```

**Making LLM Calls:**
```python
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": user_query}
    ],
    tools=tool_schemas  # Optional: for function calling
)
```

**Response Structure:**
```python
# Access output
for part in response.output:
    if part.type == "message":
        print(part.content)  # Text response
    elif part.type == "function_call":
        name = part.name  # Function name
        args = part.arguments  # JSON string arguments
        call_id = part.call_id  # Unique call identifier
```

### Sandbox

**Creating a Sandbox:**
```python
from e2b_code_interpreter import Sandbox

sbx = Sandbox.create(timeout=60 * 60)  # 1 hour timeout
```

**Running Code:**
```python
# Python code (default)
execution = sbx.run_code("print('Hello World!')")

# JavaScript code
execution = sbx.run_code("console.log('Hello')", language="javascript")

# Bash commands
execution = sbx.run_code("ls -la", language="bash")
```

**Execution Results:**
```python
# Access results
execution.results  # List of result objects
execution.error    # Error object if execution failed

# Convert to JSON
execution.to_json()  # Returns dict representation
```

**File Operations:**
```python
# Write file
sbx.files.write("/path/to/file.txt", "content")

# Read file
content = sbx.files.read("/path/to/file.txt")

# Create directory
sbx.files.make_dir("/path/to/directory")

# Remove file
sbx.files.remove("/path/to/file.txt")
```

**Listing Sandboxes:**
```python
from e2b_code_interpreter import Sandbox

running_sandboxes = Sandbox.list().next_items()
for sbx in running_sandboxes:
    print(sbx.sandbox_id)
```

**Querying Sandboxes with Metadata:**
```python
from e2b_code_interpreter import Sandbox, SandboxQuery, SandboxState

# Create sandbox with metadata
sbx = Sandbox.create(metadata={"name": "my-sandbox"})

# Query by metadata
results = Sandbox.list(SandboxQuery(
    metadata={"name": "my-sandbox"},
    state=[SandboxState.RUNNING]
))
sandboxes = results.next_items()
```

**Reconnecting to Sandbox:**
```python
sandbox_id = "sbx-abc123"
sbx = Sandbox.connect(sandbox_id)
```

**Killing a Sandbox:**
```python
sbx.kill()
```

---

## Tool Schema Pattern

**Function Schema Structure:**
```python
tool_schema = {
    "type": "function",
    "name": "function_name",
    "description": "What this function does",
    "parameters": {
        "type": "object",
        "properties": {
            "param_name": {
                "type": "string",  # or "number", "boolean", "array", "object"
                "description": "What this parameter does"
            }
        },
        "required": ["param_name"],
        "additionalProperties": False
    }
}
```

**Example: execute_code Schema:**
```python
execute_code_schema = {
    "type": "function",
    "name": "execute_code",
    "description": "Execute Python code and return the result or error.",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute as a string"
            }
        },
        "required": ["code"],
        "additionalProperties": False
    }
}
```

---

## Common Patterns

### Pattern 1: Local Code Execution (for testing)

```python
import sys
from io import StringIO

def execute_code(code: str) -> dict:
    """Execute code locally and return results/errors"""
    execution = {"results": [], "errors": []}
    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        exec(code)
        result = sys.stdout.getvalue()
        sys.stdout = old_stdout
        execution["results"] = [result]
    except Exception as e:
        execution["errors"] = [str(e)]
    finally:
        sys.stdout = old_stdout
        return execution
```

### Pattern 2: Tool Execution

```python
import json
from typing import Callable

def execute_tool(name: str, args: str, tools: dict[str, Callable]):
    """Execute a tool by name with JSON arguments"""
    try:
        args = json.loads(args)
        if name not in tools:
            return {"error": f"Tool {name} doesn't exist."}
        result = tools[name](**args)
    except json.JSONDecodeError as e:
        result = {"error": f"{name} failed to parse arguments: {str(e)}"}
    except KeyError as e:
        result = {"error": f"Missing key in arguments: {str(e)}"}
    except Exception as e:
        result = {"error": str(e)}
    return result

# Tool mapping
tools = {
    "execute_code": execute_code_function,
    "read_file": read_file_function,
    "write_file": write_file_function
}
```

### Pattern 3: Agent Loop with Max Steps

```python
def coding_agent(
    client: OpenAI,
    query: str,
    system: str,
    tools: dict[str, Callable],
    tools_schemas: list[dict],
    messages: list[dict] = None,
    max_steps: int = 5
):
    """Iterative agent that calls tools until completion or max steps"""
    if messages is None:
        messages = []

    messages.append({"role": "user", "content": query})
    steps = 0

    while steps < max_steps:
        # Call LLM
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "developer", "content": system},
                *messages
            ],
            tools=tools_schemas
        )

        has_function_call = False

        # Process response
        for part in response.output:
            messages.append(part.to_dict())

            if part.type == "message":
                print(f"[agent] {part.content}")

            elif part.type == "function_call":
                has_function_call = True
                name = part.name
                result = execute_tool(name, part.arguments, tools)

                # Add function result to messages
                messages.append({
                    "type": "function_call_output",
                    "call_id": part.call_id,
                    "output": json.dumps(result)
                })

        # Stop if no function calls
        if not has_function_call:
            break

        steps += 1

    return messages
```

### Pattern 4: Sandbox Code Execution

```python
def execute_code_in_sandbox(code: str, sbx: Sandbox) -> tuple[dict, dict]:
    """Execute code in E2B sandbox and return results + metadata"""
    metadata = {}
    execution = sbx.run_code(code)

    # Handle images (charts, plots)
    results = execution.results
    for result in results:
        if result.png:
            metadata["images"] = [result.png]
            result.png = None

    return execution.to_json(), metadata
```

### Pattern 5: Agent with Sandbox

```python
def coding_agent_sandbox(
    client: OpenAI,
    sbx: Sandbox,
    query: str,
    system: str,
    tools: dict[str, Callable],
    tools_schemas: list[dict],
    max_steps: int = 5
):
    """Agent that executes code in E2B sandbox"""
    messages = [{"role": "user", "content": query}]
    steps = 0

    while steps < max_steps:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "developer", "content": system},
                *messages
            ],
            tools=tools_schemas
        )

        has_function_call = False

        for part in response.output:
            messages.append(part.to_dict())

            if part.type == "function_call":
                has_function_call = True
                name = part.name
                # Pass sandbox to tool
                result, metadata = execute_tool(
                    name, part.arguments, tools, sbx=sbx
                )

                messages.append({
                    "type": "function_call_output",
                    "call_id": part.call_id,
                    "output": json.dumps(result)
                })

        if not has_function_call:
            break

        steps += 1

    return messages
```

---

## Best Practices

1. **Always set timeouts**: Use `timeout` parameter when creating sandboxes
2. **Handle errors gracefully**: Wrap tool execution in try-except blocks
3. **Use max_steps**: Prevent infinite loops in agent execution
4. **Clean up sandboxes**: Call `sbx.kill()` when done to free resources
5. **Use metadata for querying**: Tag sandboxes with metadata for easy retrieval
6. **Convert execution results**: Use `execution.to_json()` for consistent formatting

---

## Common Pitfalls

- **Missing tool schemas**: LLM can't call tools without proper schemas
- **Infinite loops**: Always use `max_steps` to limit agent iterations
- **Forgotten sandboxes**: Kill sandboxes when done to avoid resource waste
- **Incorrect message format**: Function call outputs must include `call_id`
- **Tool execution errors**: Always handle JSON parsing errors in `execute_tool`

---

## Example: Complete Minimal Agent

```python
from openai import OpenAI
from e2b_code_interpreter import Sandbox
import json

# Setup
client = OpenAI()
sbx = Sandbox.create(timeout=3600)

# Tool schema
execute_code_schema = {
    "type": "function",
    "name": "execute_code",
    "description": "Execute Python code",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code"}
        },
        "required": ["code"],
        "additionalProperties": False
    }
}

# Tool implementation
def execute_code(code: str, sbx: Sandbox):
    execution = sbx.run_code(code)
    return execution.to_json(), {}

tools = {"execute_code": execute_code}

# Execute tool helper
def execute_tool(name: str, args: str, tools: dict, **kwargs):
    args = json.loads(args)
    result, metadata = tools[name](**args, **kwargs)
    return result, metadata

# System prompt
system = "You are a senior Python programmer. Use execute_code to run code."

# Query
query = "Create a function that adds two numbers and test it with 5 and 3"

# Agent loop
messages = [{"role": "user", "content": query}]
for step in range(5):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{"role": "developer", "content": system}, *messages],
        tools=[execute_code_schema]
    )

    has_function_call = False
    for part in response.output:
        messages.append(part.to_dict())
        if part.type == "function_call":
            has_function_call = True
            result, _ = execute_tool(part.name, part.arguments, tools, sbx=sbx)
            messages.append({
                "type": "function_call_output",
                "call_id": part.call_id,
                "output": json.dumps(result)
            })

    if not has_function_call:
        break
```
