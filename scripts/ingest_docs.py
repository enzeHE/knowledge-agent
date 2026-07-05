"""
文档入库脚本
运行方式：python scripts/ingest_docs.py
"""
print("Script started")
from pathlib import Path
import sys
print("Imports 1 done")
sys.path.insert(0, str(Path(__file__).parent.parent))
print("Path inserted")

from app.rag.loader import load_document
print("Loader imported")
from app.rag.splitter import split_documents
print("Splitter imported")
from app.rag.retriever import retriever
print("Retriever imported")
from app.core.config import settings
print("Settings imported")
print("All imports done")


def ingest():
    print("Starting ingest process...")
    docs_dir = Path(__file__).parent.parent / "data" / "docs"

    # 获取所有 .md 文件
    md_files = list(docs_dir.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    # 创建 collection（bge-m3 的向量维度是 1024）
    print("Creating Qdrant collection...")
    # 删除旧 collection（维度变更时需要重建）
    try:
        retriever._get_client().delete_collection(settings.qdrant_collection)
    except Exception:
        pass
    retriever.create_collection(vector_size=1024)
    print("Collection created.")

    all_chunks = []
    all_metadata = []

    for i, file_path in enumerate(md_files):
        print(f"[{i+1}/{len(md_files)}] Processing {file_path.name}...")

        try:
            docs = load_document(str(file_path))
            chunks = split_documents(docs)

            for chunk in chunks:
                all_chunks.append(chunk.page_content)
                all_metadata.append({
                    "source": str(file_path.relative_to(docs_dir)),
                    "filename": file_path.name
                })
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Embedding and indexing...")

    # 批量入库
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[i:i+batch_size]
        batch_meta = all_metadata[i:i+batch_size]
        retriever.add_documents(batch_texts, batch_meta)
        print(f"  Indexed {min(i+batch_size, len(all_chunks))}/{len(all_chunks)}")

    print("\nDone! Test with:")
    print('  python -c "from app.rag.retriever import retriever; print(retriever.hybrid_search(\'fastapi query parameters\'))"')


if __name__ == "__main__":
    ingest()
