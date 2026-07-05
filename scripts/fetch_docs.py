"""
运行方式：python scripts/fetch_docs.py
将三个官方文档仓库的 .md 文件复制到 data/docs/
"""
import os
import shutil
import subprocess
from pathlib import Path

REPOS = [
    {
        "url": "https://github.com/langchain-ai/langchain",
        "docs_path": "libs/langchain/docs",
        "target": "langchain",
    },
    {
        "url": "https://github.com/langchain-ai/langgraph",
        "docs_path": "docs",
        "target": "langgraph",
    },
    {
        "url": "https://github.com/fastapi/fastapi",
        "docs_path": "docs/en/docs",
        "target": "fastapi",
    },
]

BASE_DIR = Path(__file__).parent.parent
TMP_DIR = BASE_DIR / "data" / "tmp"
DOCS_DIR = BASE_DIR / "data" / "docs"


def fetch():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    for repo in REPOS:
        clone_dir = TMP_DIR / repo["target"]
        target_dir = DOCS_DIR / repo["target"]

        print(f"Cloning {repo['target']}...")
        if not clone_dir.exists():
            subprocess.run(
                ["git", "clone", "--depth=1", repo["url"], str(clone_dir)],
                check=True,
            )

        src = clone_dir / repo["docs_path"]
        if src.exists():
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(src, target_dir, ignore=shutil.ignore_patterns("*.py", "*.js", "*.css"))
            md_count = len(list(target_dir.rglob("*.md")))
            print(f"  {repo['target']}: {md_count} .md files copied to {target_dir}")
        else:
            print(f"  WARNING: {src} not found, check docs_path config")


if __name__ == "__main__":
    fetch()
    print("\nDone. Run the ingest pipeline next.")
