import json
from typing import Callable, Optional
from e2b_code_interpreter import Execution, Sandbox


def execute_code(sbx: Sandbox, code: str, language: str = "python") -> Execution:
    execution = sbx.run_code(code, language)
    metadata = {}
    results = execution.results
    for result in results:
        if result.png:
            metadata["images"] = [result.png]
            result.png = None
            result.chart = None
    return execution.to_json(), metadata


tools = {
    "execute_code": execute_code,
    "execute_bash": lambda **a: execute_code(**a, language="bash"),
    "list_directory": lambda **a: execute_code(
        a["sbx"],
        f"list_directory(secure_path({repr(a.get('path', '.'))}), {repr(a.get('ignore'))}, {a.get('offset', 0)}, {a.get('limit', 16)})",
    ),
    "read_file": lambda **a: execute_code(
        a["sbx"],
        f"read_file(secure_path({repr(a.get('file_path', ''))}), {a.get('limit')}, {a.get('offset', 0)})",
    ),
    "write_file": lambda **a: execute_code(
        a["sbx"],
        f"write_file({repr(a.get('content', ''))}, secure_path({repr(a.get('file_path', ''))}))",
    ),
    "replace_in_file": lambda **a: execute_code(
        a["sbx"],
        f"replace_in_file(secure_path({repr(a.get('file_path', ''))}), {repr(a.get('old_string', ''))}, {repr(a.get('new_string', ''))}, {a.get('expected_replacements', 1)})",
    ),
    "search_file_content": lambda **a: execute_code(
        a["sbx"],
        f"search_file_content({repr(a.get('pattern', ''))}, {repr(a.get('include'))}, secure_path({repr(a.get('path', '.'))}), {a.get('use_regex', False)}, {a.get('fuzzy_threshold')}, {a.get('offset', 0)}, {a.get('limit', 16)})",
    ),
    "glob": lambda **a: execute_code(
        a["sbx"],
        f"glob({repr(a.get('pattern', ''))}, secure_path({repr(a.get('path', '.'))}), {repr(a.get('ignore'))}, {a.get('offset', 0)}, {a.get('limit', 16)})",
    ),
}


def execute_tool(name: str, args: str, tools: dict[str, Callable], **kwargs):
    metadata = {}
    try:
        args = json.loads(args)
        if name not in tools:
            return {"error": f"Tool {name} doesn't exist."}
        result, metadata = tools[name](**args, **kwargs)
    except json.JSONDecodeError as e:
        result = {"error": f"{name} failed to parse arguments: {str(e)}"}
    except KeyError as e:
        result = {"error": f"Missing key in arguments: {str(e)}"}
    except Exception as e:
        result = {"error": str(e)}
    return result, metadata
