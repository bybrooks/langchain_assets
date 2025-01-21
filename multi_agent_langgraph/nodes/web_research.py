from typing import Literal

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.types import Command

from ..utils import create_agent, create_llm_model

llm = create_llm_model()
web_search_tool = TavilySearchResults(max_results=2)
websearch_agent = create_agent(llm, [web_search_tool])


def web_research_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = websearch_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="web_researcher"
                )
            ]
        },
        goto="supervisor",
    )
