import json
import sys
import os
import traceback

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    # 设置工作目录
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    os.environ["HF_HUB_OFFLINE"] = "1"

    from app.core.config import settings
    from app.rag.retriever import retriever
    from app.rag.embedder import embedder
    from langchain_anthropic import ChatAnthropic
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset

    print("All imports successful")

    llm = ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    wrapped_llm = LangchainLLMWrapper(llm)


    class SimpleEmbeddings:
        def embed_documents(self, texts):
            return embedder.embed_batch(texts)

        def embed_query(self, text):
            return embedder.embed_text(text)


    wrapped_embeddings = LangchainEmbeddingsWrapper(SimpleEmbeddings())

    # 检查 Qdrant 是否可用
    qdrant_available = False
    try:
        client = retriever._get_client()
        client.get_collections()
        qdrant_available = True
        print(f"Qdrant connected: {settings.qdrant_url}")
    except Exception as e:
        print(f"Qdrant not available ({e}), will use empty contexts for retrieval")

    with open(os.path.join(os.path.dirname(__file__), "dataset.json")) as f:
        dataset = json.load(f)

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    retrieval_errors = []

    print(f"\nRunning inference on {len(dataset)} questions...")
    if qdrant_available:
        print("(First query will trigger embedding model & BM25 index loading...)")
    else:
        print("(No Qdrant — LLM will answer from its own knowledge; RAGAS scores will reflect missing retrieval)")

    for i, item in enumerate(dataset):
        q = item["question"]
        context_texts = []
        retrieval_ok = False

        if qdrant_available:
            try:
                docs = retriever.hybrid_search(q, top_k=3)
                context_texts = [d["text"] for d in docs]
                retrieval_ok = bool(context_texts)
            except Exception as e:
                retrieval_errors.append((q, str(e)))
                # 兜底：嵌入模型预热
                try:
                    embedder.embed_text(q)
                except Exception:
                    pass

        context_str = "\n\n".join(context_texts) if context_texts else "No relevant context available."
        instruction = (
            f"Answer the question based on the context below.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {q}"
        )

        response = llm.invoke(instruction)
        if isinstance(response.content, str):
            answer = response.content
        elif isinstance(response.content, list):
            answer = "".join(
                block.get("text", "") for block in response.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            answer = str(response.content)

        rows["question"].append(q)
        rows["answer"].append(answer)
        rows["contexts"].append(context_texts)
        rows["ground_truth"].append(item["ground_truth"])

        status = f"  [{i+1}/{len(dataset)}] {'retrieval OK' if retrieval_ok else 'no context'}"
        print(status)

    # 汇总检索错误
    if retrieval_errors:
        print(f"\nRetrieval errors ({len(retrieval_errors)}):")
        for q, err in retrieval_errors[:5]:
            print(f"  - {q[:60]}: {err[:100]}")

    print("\nRunning RAGAS evaluation...")
    result = evaluate(
        Dataset.from_dict(rows),
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=wrapped_llm,
        embeddings=wrapped_embeddings,
    )

    print("\n=== RAGAS Results ===")
    print(result)

    # 导出详细结果
    df = result.to_pandas()
    df["retrieval_available"] = qdrant_available

    output_path = os.path.join(os.path.dirname(__file__), "ragas_results.json")
    df.to_json(output_path, orient="records", indent=2, force_ascii=False)
    print(f"\nDetailed results saved to {output_path}")

    # 输出汇总
    print("\n=== Summary ===")
    print(f"Total questions: {len(dataset)}")
    print(f"Qdrant available: {qdrant_available}")
    print(f"FastAPI questions: 26")
    print(f"LangChain questions: 20")
    print(f"LangGraph questions: 14")
    print(f"\nAverage faithfulness: {df['faithfulness'].mean():.3f}")
    print(f"Average answer_relevancy: {df['answer_relevancy'].mean():.3f}")
    print(f"Average context_precision: {df['context_precision'].mean():.3f}")

except Exception as e:
    print(f"\n!!! ERROR: {e}")
    traceback.print_exc()
