import requests
import json

def chat(message, thread_id):
    url = "http://localhost:8000/api/chat"
    data = {"message": message, "thread_id": thread_id}
    response = requests.post(url, json=data, stream=True)
    print(f"\n[User]: {message}")
    print("[Agent]: ", end="")
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                content = line[6:]
                if content != "[DONE]":
                    chunk = json.loads(content)
                    print(chunk.get("content", ""), end="", flush=True)
    print()

# 第一轮对话
chat("My name is Alex. What is dependency injection in FastAPI?", "memory-test-1")

# 第二轮对话（新 thread，测试跨会话记忆）
print("\n--- 第二轮（新 thread，应无记忆）---")
chat("What is my name?", "memory-test-2")

# 第三轮对话（同 thread，测试短期记忆）
print("\n--- 第三轮（同 thread，应记得名字）---")
chat("What is my name?", "memory-test-1")
