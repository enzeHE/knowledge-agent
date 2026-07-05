import asyncio
import os
import sys
sys.path.insert(0, "D:/knowledge-agent")
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_core.messages import AIMessage
from app.agent.graph import agent_graph, build_initial_messages


async def main():
    initial_messages = build_initial_messages("", "What is a path parameter in FastAPI?")
    config = {"configurable": {"thread_id": "astream-test-2"}}

    try:
        async for event in agent_graph.astream(
            {"messages": initial_messages},
            config,
            stream_mode="values"
        ):
            last = event["messages"][-1]
            print(f"[{type(last).__name__}]", repr(str(last.content)[:80]))
        print("Done.")
    except Exception as e:
        import traceback
        traceback.print_exc()


asyncio.run(main())
