import sys
sys.path.insert(0, "D:/knowledge-agent")

from app.agent.graph import agent_graph

config = {"configurable": {"thread_id": "test-1"}}

print("Testing Agent with RAG tool...\n")

query = "How do I use query parameters in FastAPI?"
print(f"User: {query}\n")

for event in agent_graph.stream(
    {"messages": [("user", query)]},
    config,
    stream_mode="values"
):
    event["messages"][-1].pretty_print()
