from pathlib import Path
from typing import List
from langchain.schema import Document


def load_document(file_path: str) -> List[Document]:
    """加载单个 Markdown 文件"""
    path = Path(file_path)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return [Document(
        page_content=content,
        metadata={"source": str(path.name), "filepath": str(path)}
    )]


def load_documents_from_dir(directory: str, pattern: str = "*.md") -> List[Document]:
    """递归加载目录下所有 Markdown 文件"""
    dir_path = Path(directory)
    docs = []
    for file_path in sorted(dir_path.rglob(pattern)):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            rel_path = file_path.relative_to(dir_path)
            docs.append(Document(
                page_content=content,
                metadata={"source": str(rel_path), "filepath": str(file_path)}
            ))
        except Exception as e:
            print(f"  Skipping {file_path.name}: {e}")
            continue
    return docs
