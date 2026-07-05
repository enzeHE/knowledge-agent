import gradio as gr
import requests
import json
import time

API_BASE = "http://localhost:8001/api"


def chat_with_agent(message, history, thread_id):
    if not message.strip():
        return history, []

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    try:
        response = requests.post(
            f"{API_BASE}/chat",
            json={"message": message, "thread_id": thread_id},
            stream=True,
            timeout=60
        )

        full_response = ""
        contexts = []
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if data.get("type") == "text":
                            full_response += data.get("content", "")
                            history[-1]["content"] = full_response
                            yield list(history), contexts
                        elif data.get("type") == "contexts":
                            contexts = parse_contexts(data.get("content", []))
                            yield list(history), contexts
                    except:
                        continue

        yield list(history), contexts

    except Exception as e:
        history[-1]["content"] = f"Error: {str(e)}"
        yield list(history), []


def parse_contexts(raw_contexts):
    """解析检索结果，提取来源和文本"""
    sources = []
    for ctx in raw_contexts:
        # ctx 是 ToolMessage.content，格式如 "[1] text...\nSource: path\n[2]..."
        lines = ctx.split('\n')
        current_text = ""
        current_source = ""

        for line in lines:
            if line.startswith("Source: "):
                current_source = line[8:].strip()
                if current_text:
                    sources.append({"source": current_source, "text": current_text.strip()})
                    current_text = ""
            elif line.strip():
                current_text += line + " "

        if current_text and current_source:
            sources.append({"source": current_source, "text": current_text.strip()})

    return sources


def upload_document(file):
    """上传文档到后端"""
    if file is None:
        return "请选择文件", "", ""

    try:
        with open(file.name, 'rb') as f:
            files = {'file': (file.name.split('/')[-1], f)}
            response = requests.post(f"{API_BASE}/documents/upload", files=files)
            result = response.json()

        doc_id = result.get("doc_id")
        status_msg = f"上传成功！文档ID: {doc_id}, 状态: {result.get('status')}"

        # 轮询状态（后台运行，不阻塞）
        for _ in range(30):
            time.sleep(2)
            status_resp = requests.get(f"{API_BASE}/documents/{doc_id}")
            status_data = status_resp.json()
            if status_data.get("status") in ("completed", "done"):
                process_msg = f"✅ 处理完成，共 {status_data.get('chunk_count', 0)} 个chunks"
                break
            elif status_data.get("status") == "failed":
                process_msg = "❌ 处理失败"
                break
        else:
            process_msg = "⏳ 处理中..."

        # 刷新文档列表
        doc_list = fetch_document_list()
        return status_msg, process_msg, doc_list

    except Exception as e:
        return f"上传失败: {str(e)}", "", ""


def fetch_document_list():
    """获取文档列表"""
    try:
        resp = requests.get(f"{API_BASE}/documents/", timeout=10)
        docs = resp.json()
        if not docs:
            return []
        return [[d.get("doc_id"), d.get("filename"), d.get("status"), d.get("chunk_count")] for d in docs]
    except Exception:
        return []


# 构建 UI
with gr.Blocks(title="Knowledge Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 Knowledge Agent - FastAPI 文档助手")

    with gr.Tabs():
        # Tab 1: 对话
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=500, label="对话历史", type="messages")
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="输入你的问题，例如：How to define path parameters in FastAPI?",
                            show_label=False,
                            scale=4
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1)

                    thread_input = gr.Textbox(
                        value="default",
                        label="Thread ID (多轮对话标识)",
                        interactive=True
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📎 引用来源")
                    sources_display = gr.JSON(label="检索到的文档片段", value=[])

            send_btn.click(
                chat_with_agent,
                inputs=[msg_input, chatbot, thread_input],
                outputs=[chatbot, sources_display]
            )
            msg_input.submit(
                chat_with_agent,
                inputs=[msg_input, chatbot, thread_input],
                outputs=[chatbot, sources_display]
            )

        # Tab 2: 文档管理
        with gr.Tab("📂 Documents"):
            gr.Markdown("### 上传文档到知识库")
            with gr.Row():
                file_input = gr.File(label="选择 Markdown 文件", file_types=[".md"])
                upload_btn = gr.Button("上传", variant="primary")

            upload_status = gr.Textbox(label="上传状态", interactive=False)
            process_status = gr.Textbox(label="处理状态", interactive=False)

            gr.Markdown("### 文档列表")
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新列表", variant="secondary")
            doc_table = gr.Dataframe(
                headers=["ID", "文件名", "状态", "Chunks"],
                label="已入库文档",
                value=[],
                row_count=10,
            )

            upload_btn.click(
                upload_document,
                inputs=[file_input],
                outputs=[upload_status, process_status, doc_table]
            )
            refresh_btn.click(
                fetch_document_list,
                inputs=[],
                outputs=[doc_table]
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
