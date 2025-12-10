from e2b_code_interpreter import Sandbox
from e2b_code_interpreter import SandboxQuery, SandboxState
from uuid import uuid4
from .logger import logger
from pathlib import Path

CACHE_FILENAME = "sbx.cache"

NEXTJS_TEMPLATE_ID = "dlai-nextjs-developer"


def setup_sandbox(sbx: Sandbox):
    packages = ["rapidfuzz"]
    entries = sbx.files.list("")
    # we go to the working dir
    has_file = list(filter(lambda entry: entry.name == "sbx_tools.py", entries))
    if has_file:
        return
    # install packages
    logger.info("[sandbox] 🔧 Setting it up ...")
    logger.info("[sandbox] \tinstalling packages ...")
    sbx.run_code(f"pip install {' '.join(packages)}", language="bash")
    # copy our .py file
    with open("lib/sbx_tools.py", "r") as f:
        content = f.read()
    logger.info("[sandbox] \tcopying tools ...")
    sbx.files.write("sbx_tools.py", content)
    sbx.run_code("from sbx_tools import *")
    logger.info("[sandbox] 🔧 done!")


def create_sandbox(template: str = None, overwrite: bool = False, **kwargs) -> Sandbox:
    cache_file = Path(CACHE_FILENAME)

    if cache_file.exists():
        name = cache_file.read_text()
    else:
        name = f"dlai-sbx-{template}-{uuid4()}"
        cache_file.write_text(name)

    if not overwrite:
        running_sandboxes = Sandbox.list(
            SandboxQuery(metadata={"name": name}, state=[SandboxState.RUNNING])
        ).next_items()
        if running_sandboxes:
            sandbox = Sandbox.connect(running_sandboxes[0].sandbox_id)
            logger.info(
                f"[sandbox] 🔌 Reconnecting to Sandbox.create(id={sandbox.sandbox_id})"
            )
            return sandbox

    sandbox = Sandbox.create(
        timeout=60 * 60, metadata={"name": name}, template=template, **kwargs
    )
    logger.info(f"[sandbox] 🚀 Creating new Sandbox.create(id={sandbox.sandbox_id})")
    if template == NEXTJS_TEMPLATE_ID:
        setup_sandbox(sandbox)

    return sandbox


def clear_sandboxes():
    paginator = Sandbox.list(SandboxQuery(state=[SandboxState.RUNNING]))
    try:
        while sandboxes := paginator.next_items():
            for sandbox in sandboxes:
                Sandbox.connect(sandbox.sandbox_id).kill()
                logger.info(f"[sandbox]  Killed  Sandbox(id={sandbox.sandbox_id})")
    except Exception:
        pass
