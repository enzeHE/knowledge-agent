from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.schema import Document
from typing import List
import tiktoken

# cl100k_base tokenizer，和 LLM 使用的 tokenizer 一致
_tokenizer = tiktoken.get_encoding("cl100k_base")


def token_len(text: str) -> int:
    return len(_tokenizer.encode(text))


def split_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    两级分块策略：
    1. MarkdownHeaderTextSplitter — 按 H1/H2/H3 标题层级分割，保证语义完整
    2. RecursiveCharacterTextSplitter — 对大段在段落边界二次切分，控制 token 长度
    """
    # Stage 1: 按 Markdown 标题层级分割
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
        ],
    )

    header_chunks: List[Document] = []
    for doc in documents:
        splits = header_splitter.split_text(doc.page_content)
        for chunk in splits:
            # 合并原始元数据 + 标题层级元数据
            chunk.metadata = {**doc.metadata, **chunk.metadata}
        header_chunks.extend(splits)

    # Stage 2: 对超长 chunk 做二次切分
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_len,
        separators=["\n\n", "\n", " ", ""],
    )

    final_chunks = recursive_splitter.split_documents(header_chunks)
    return final_chunks
