import requests
import json

url = "http://localhost:8000/api/chat"
data = {
    "message": "How to use path parameters in FastAPI?",
    "thread_id": "test-1"
}

response = requests.post(url, json=data, stream=True, timeout=120)

print("Response:")
for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            content = line[6:]
            if content == '[DONE]':
                print("\n[DONE]")
            else:
                chunk = json.loads(content)
                if chunk.get('type') == 'error':
                    print(f"\n[ERROR]: {chunk.get('content')}")
                else:
                    print(chunk.get('content', ''), end='', flush=True)

print("\nDone!")
