from typing import Any, Literal

from langgraph.graph import END, MessagesState
from langgraph.types import Command
from pydantic import BaseModel

from ..utils import create_llm_model

members = ["web_researcher", "rag", "nl2sql"]
# Our supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)


class Router(BaseModel):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal["web_researcher", "rag", "nl2sql", "FINISH"]


def supervisor_node(
    state: MessagesState,
) -> Command[Literal["web_researcher", "rag", "nRouterl2sql", "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    llm = create_llm_model()
    response: dict[Any, Any] | BaseModel = llm.with_structured_output(Router).invoke(
        messages
    )

    match type(response):
        case dict():
            RuntimeError("Structured output not returned")
        case BaseModel():
            pass
        case _:
            raise ValueError("Invalid type")

    goto = response.next
    print(f"Next Worker: {goto}")
    if goto == "FINISH":
        goto = END

    return Command(goto=goto)
