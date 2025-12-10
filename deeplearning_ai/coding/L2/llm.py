from openai import OpenAI
from openai.types.responses import Response


def llm(
    client: OpenAI,
    messages: list[dict],
    system: str = "You are an helpful assistant",
    name: str = "gpt-4.1-mini",
    **kwargs,
) -> Response:
    system_message = {"role": "developer", "content": system}
    if name.startswith("gpt-4"):
        kwargs["temperature"] = 0
    response = client.responses.create(
        model=name,
        input=[system_message, *messages],
        **kwargs,
    )
    return response
