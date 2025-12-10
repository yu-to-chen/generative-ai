"""
Sandbox Tools - File system operations for e2b sandbox
Clean, fast, throws proper exceptions.
"""

import os
import fnmatch
import glob as glob_module
import re
from typing import Dict, List, Any, Optional


class ToolError(Exception):
    """Custom exception for tool failures - gets caught by e2b execution"""

    pass


def _paginate_results(
    results: List[Any], offset: int = 0, limit: Optional[int] = 16
) -> Dict[str, Any]:
    """Slice results and return pagination metadata"""
    total = len(results)
    start = max(0, offset)
    limit = min(limit, 64)
    end = start + limit if limit else total
    page = results[start:end]

    return {
        "pagination": {
            "total": total,
            "offset": start,
            "limit": limit,
            "has_more": end < total,
        },
        "results": page[start:end],
    }


def secure_path(requested_path: str) -> str:
    """Keep paths locked to working_dir or die trying"""
    working_dir = os.getcwd()
    wd_real = os.path.realpath(working_dir)

    if not requested_path:
        return wd_real

    # Handle absolute vs relative paths
    if os.path.isabs(requested_path):
        target_real = os.path.realpath(requested_path)
    else:
        target_real = os.path.realpath(os.path.join(wd_real, requested_path))

    # Ensure target is within working_dir
    if not target_real.startswith(wd_real + os.sep) and target_real != wd_real:
        raise ToolError(
            f"Path '{requested_path}' escapes working directory. You can read/edit only files in '{working_dir}.'"
        )

    return target_real


def list_directory(
    path: str = ".",
    ignore: Optional[List[str]] = None,
    offset: int = 0,
    limit: Optional[int] = 16,
) -> Dict[str, Any]:
    """List directory contents with pagination."""
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise ToolError(f"Path does not exist: {path}")
    if not os.path.isdir(path):
        raise ToolError(f"Path is not a directory: {path}")

    entries = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if ignore and any(fnmatch.fnmatch(item, pattern) for pattern in ignore):
            continue

        stat = os.stat(item_path)
        entries.append(
            {
                "name": item,
                "type": "directory" if os.path.isdir(item_path) else "file",
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
        )

    entries.sort(key=lambda x: (x["type"] != "directory", x["name"]))
    return {**_paginate_results(entries, offset, limit), "path": path}


def read_file(
    file_path: str, limit: Optional[int] = 16, offset: int = 0
) -> Dict[str, Any]:
    """Read file content with optional offset and limit."""
    if not os.path.exists(file_path):
        raise ToolError(f"File does not exist: {file_path}")

    if not os.path.isfile(file_path):
        raise ToolError(f"Path is not a file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if offset > 0:
                f.seek(offset)
            content = f.read(limit) if limit else f.read()

        return {"content": content, "size": len(content)}

    except PermissionError:
        raise ToolError(f"Permission denied: {file_path}")
    except UnicodeDecodeError:
        raise ToolError(f"Cannot decode file as UTF-8: {file_path}")


def write_file(content: str, file_path: str) -> Dict[str, Any]:
    """Write content to file, creating directories if needed."""
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        file_size = os.path.getsize(file_path)
        return {
            "message": f"Written {file_size} bytes to {file_path}",
            "size": file_size,
        }

    except PermissionError:
        raise ToolError(f"Permission denied: {file_path}")


def replace_in_file(
    file_path: str, old_string: str, new_string: str, expected_replacements: int = 1
) -> Dict[str, Any]:
    """Replace text in file with validation."""
    if not os.path.exists(file_path):
        raise ToolError(f"File does not exist: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    actual_count = content.count(old_string)
    if actual_count != expected_replacements:
        raise ToolError(
            f"Expected {expected_replacements} occurrences, found {actual_count}"
        )

    new_content = content.replace(old_string, new_string, expected_replacements)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return {
        "replacements": expected_replacements,
        "message": f"Replaced {expected_replacements} occurrences",
    }


def search_file_content(
    pattern: str,
    include: Optional[str] = None,
    path: str = ".",
    use_regex: bool = False,
    fuzzy_threshold: Optional[int] = None,
    offset: int = 0,
    limit: Optional[int] = 16,
) -> Dict[str, Any]:
    """Search for pattern in file contents with pagination."""
    results = []
    total_files_searched = 0

    regex_pattern = None
    if use_regex:
        try:
            regex_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise ToolError(f"Invalid regex pattern: {e}")

    for root, dirs, files in os.walk(path):
        for file in files:
            if include and not fnmatch.fnmatch(file, include):
                continue

            filepath = os.path.join(root, file)
            total_files_searched += 1

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        line_stripped = line.strip()
                        match_data = {
                            "file": filepath,
                            "line": line_num,
                            "content": line_stripped,
                        }

                        if fuzzy_threshold is not None:
                            similarity = fuzz.partial_ratio(
                                pattern.lower(), line.lower()
                            )
                            if similarity >= fuzzy_threshold:
                                match_data["similarity"] = similarity
                                results.append(match_data)
                        elif use_regex:
                            if regex_pattern.search(line):
                                results.append(match_data)
                        else:
                            if pattern.lower() in line.lower():
                                results.append(match_data)
            except:
                continue

    if fuzzy_threshold is not None:
        results.sort(key=lambda x: x["similarity"], reverse=True)

    paginated = _paginate_results(results, offset, limit)
    return {**paginated, "files_searched": total_files_searched}


def glob(
    pattern: str,
    path: str = ".",
    ignore: Optional[List[str]] = None,
    offset: int = 0,
    limit: Optional[int] = 16,
) -> Dict[str, Any]:
    """Find files matching glob pattern with pagination and ignore patterns."""
    original_cwd = os.getcwd()

    try:
        os.chdir(path)

        # Auto-load simple .gitignore patterns if ignore is None
        if ignore is None:
            gitignore_path = os.path.join(path, ".gitignore")
            if os.path.isfile(gitignore_path):
                ignore = []
                try:
                    with open(gitignore_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if (
                                line
                                and not line.startswith("#")
                                and not line.startswith("/")
                            ):
                                if line.endswith("/"):
                                    line = line[:-1]
                                ignore.append(line)
                except:
                    ignore = None

        matches = glob_module.glob(pattern, recursive=True)

        results = []
        for match in matches:
            if ignore:
                should_ignore = False
                path_parts = match.split(os.sep)

                for ignore_pattern in ignore:
                    # Check if any directory component matches the pattern
                    if any(
                        fnmatch.fnmatch(part, ignore_pattern) for part in path_parts
                    ):
                        should_ignore = True
                        break
                    # Check if path starts with pattern (for directory patterns)
                    if (
                        match.startswith(ignore_pattern + os.sep)
                        or match == ignore_pattern
                    ):
                        should_ignore = True
                        break
                    # Check full path pattern match
                    if fnmatch.fnmatch(match, ignore_pattern):
                        should_ignore = True
                        break

                if should_ignore:
                    continue

            abs_path = os.path.abspath(match)
            if os.path.isfile(abs_path):
                stat = os.stat(abs_path)
                results.append(
                    {
                        "path": abs_path,
                        "relative_path": match,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    }
                )

        results.sort(key=lambda x: x["modified"], reverse=True)
        return {**_paginate_results(results, offset, limit), "pattern": pattern}

    finally:
        os.chdir(original_cwd)
