import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # We'll handle missing tavily gracefully

logger = logging.getLogger("demo")
logging.basicConfig(level=logging.INFO)


def _get_tavily_api_key() -> Optional[str]:
    """
    Get Tavily API key from environment. Log if present / missing.
    We don't strictly require it here (the tools / workflow may handle it),
    but we keep the behavior similar to your previous code.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key:
        logger.info("Using Tavily API key from environment")
    else:
        logger.warning("TAVILY_API_KEY not set; Tavily-based tools may fail at runtime.")
    return api_key


class ResearchDemo:
    """
    Simple wrapper around your LangGraph `workflow_app` and optional `semantic_cache`.

    It builds a Gradio chat UI that:
    - Takes a user question
    - Optionally passes chat history
    - Calls `workflow_app.invoke(...)`
    - Displays the answer
    """

    def __init__(
        self,
        workflow_app: Any,
        semantic_cache: Optional[Any] = None,
        tavily_api_key: Optional[str] = None,
    ) -> None:
        self.workflow_app = workflow_app
        self.semantic_cache = semantic_cache

        # Tavily is optional here. The actual tools in your workflow_app
        # probably read TAVILY_API_KEY from the environment.
        self.tavily_api_key = tavily_api_key or _get_tavily_api_key()

        if TavilyClient is not None and self.tavily_api_key:
            # Use the simple, supported signature for your version:
            # TavilyClient(api_key=...)
            try:
                self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
            except TypeError:
                # If the installed `tavily` version changes signature again, just ignore.
                logger.warning(
                    "TavilyClient signature mismatch; continuing without a direct TavilyClient instance."
                )
                self.tavily_client = None
        else:
            self.tavily_client = None

    # -----------------------------------------------------
    # Core logic: how we call the workflow
    # -----------------------------------------------------
    def _run_workflow(
        self,
        question: str,
        chat_history: List[Tuple[str, str]],
    ) -> Tuple[str, str]:
        """
        Call your LangGraph workflow_app with a simple input structure.
        Adjust here if your graph expects a different shape.
        """
        if not question.strip():
            return "", "Please enter a question."

        # Basic state; modify if your graph uses different keys
        state: Dict[str, Any] = {
            "question": question,
        }

        # Pass chat history if your workflow uses it
        if chat_history:
            state["chat_history"] = [
                {"user": u, "assistant": a} for u, a in chat_history
            ]

        try:
            result = self.workflow_app.invoke(state)

            # Common pattern: result is a dict with "answer" key
            if isinstance(result, dict):
                answer = result.get("answer") or result.get("response") or str(result)
                debug_str = str(result)
            else:
                answer = str(result)
                debug_str = answer

            return answer, debug_str
        except Exception as e:
            logger.exception("Error while invoking workflow_app")
            return f"Error while running workflow: {e}", ""

    # -----------------------------------------------------
    # Gradio UI wiring
    # -----------------------------------------------------
    def build_gradio_app(self) -> gr.Blocks:
        with gr.Blocks(title="Deep Research Agent Demo") as demo:
            gr.Markdown("# Deep Research Agent Demo")

            chatbot = gr.Chatbot(height=600, label="Conversation")
            question_box = gr.Textbox(
                label="Ask a question",
                placeholder="Type your question here and press Enter...",
                lines=3,
            )
            debug_box = gr.Textbox(
                label="Debug / Raw result (optional)",
                lines=10,
                interactive=False,
                visible=False,
            )
            clear_btn = gr.Button("Clear")

            def _respond(message: str, history: List[Tuple[str, str]]):
                answer, debug_str = self._run_workflow(message, history or [])
                history = history or []
                history.append((message, answer))
                return "", history, debug_str

            question_box.submit(
                _respond,
                inputs=[question_box, chatbot],
                outputs=[question_box, chatbot, debug_box],
            )

            clear_btn.click(
                lambda: ([], "", ""),
                inputs=None,
                outputs=[chatbot, question_box, debug_box],
            )

        return demo


# ---------------------------------------------------------
# Public factory functions for __init__.py imports
# ---------------------------------------------------------
def create_demo(
    workflow_app: Any,
    semantic_cache: Optional[Any],
    tavily_api_key: Optional[str] = None,
) -> ResearchDemo:
    """
    Create and return a ResearchDemo instance.
    """
    if tavily_api_key is None:
        tavily_api_key = _get_tavily_api_key()
    demo = ResearchDemo(
        workflow_app=workflow_app,
        semantic_cache=semantic_cache,
        tavily_api_key=tavily_api_key,
    )
    return demo


def launch_demo(
    workflow_app: Any,
    semantic_cache: Optional[Any],
    tavily_api_key: Optional[str] = None,
    **kwargs: Any,
):
    """
    Launch the Gradio demo.

    Typical notebook usage:
        launch_demo(
            workflow_app,
            cache,
            share=True,
            height=1500,
            inline=True,
        )
    """
    demo_obj = create_demo(workflow_app, semantic_cache, tavily_api_key)
    app = demo_obj.build_gradio_app()

    # Respect caller kwargs; keep defaults sensible
    share = kwargs.pop("share", False)
    inline = kwargs.pop("inline", True)
    height = kwargs.pop("height", 800)

    # Gradio's launch supports `inline` and `height` in notebooks.
    return app.launch(share=share, inline=inline, height=height, **kwargs)

