from langgraph.graph import START, MessagesState, StateGraph

from .nodes import nl2sql_node, rag_node, supervisor_node, web_research_node

builder = StateGraph(MessagesState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("web_researcher", web_research_node)
builder.add_node("rag", rag_node)
builder.add_node("nl2sql", nl2sql_node)
graph = builder.compile()

for s in graph.stream(
    {"messages": [("user", "what is the current weather in Kolkata")]}, subgraphs=True
):
    print(s)
    print("----")
