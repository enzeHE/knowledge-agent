"""输入/输出护栏

输入侧：空 query 拦截 → 长度校验 → 敏感词检测
输出侧：来源一致性校验 → 来源链接追加
"""

import re

# ── 敏感词库 ──────────────────────────────────────────────
# 对技术文档场景做基本安全过滤，可按需增删

_SENSITIVE_WORDS = [
    # 中文
    "赌博", "色情", "毒品", "枪械", "爆炸物", "诈骗",
    "翻墙", "VPN 违法", "黑客教程", "密码破解",
    "出售个人信息", "银行卡号", "身份证号生成",
    "代写论文", "代考", "作弊",
    # English
    "hack", "crack", "exploit", "malware", "ransomware",
    "keygen", "pirate", "illegal", "drug", "weapon",
    "bomb", "terrorist", "phishing",
    # Prompt injection attempt
    "ignore previous instructions", "ignore all instructions",
    "ignore system prompt", "forget your instructions",
    "你不需要遵守", "忽略系统", "忽略指令",
]

# 编译正则：忽略大小写，对整个 query 做子串匹配
_SENSITIVE_PATTERN = re.compile(
    "|".join(re.escape(w) for w in _SENSITIVE_WORDS),
    re.IGNORECASE,
)

MAX_QUERY_LENGTH = 500


class InputGuardResult:
    """输入检查结果"""

    def __init__(self, passed: bool, reason: str = "", sanitized: str = ""):
        self.passed = passed
        self.reason = reason
        self.sanitized = sanitized  # 修正后的 query（例如截断后）


class InputGuard:
    """输入护栏"""

    @staticmethod
    def check(query: str) -> InputGuardResult:
        """检查用户输入

        按顺序：空 → 长度 → 敏感词
        """
        # 1. 空 query
        if not query or not query.strip():
            return InputGuardResult(
                passed=False,
                reason="Please enter a question or query.",
            )

        # 2. 长度校验
        if len(query) > MAX_QUERY_LENGTH:
            sanitized = query[:MAX_QUERY_LENGTH]
            return InputGuardResult(
                passed=True,
                reason=f"Query truncated to {MAX_QUERY_LENGTH} characters.",
                sanitized=sanitized,
            )

        # 3. 敏感词检测
        match = _SENSITIVE_PATTERN.search(query)
        if match:
            return InputGuardResult(
                passed=False,
                reason="Query contains content that cannot be processed.",
            )

        return InputGuardResult(passed=True)


# ── 输出护栏 ──────────────────────────────────────────────

# 英文停用词（不计为需要验证的关键实体）
_STOP_WORDS = {
    "the", "this", "that", "these", "those", "and", "for", "are",
    "can", "you", "will", "with", "from", "your", "its", "has",
    "have", "been", "was", "were", "would", "could", "should",
    "about", "than", "then", "into", "over", "also", "very",
    "just", "each", "use", "using", "used", "how", "why",
    "what", "when", "where", "which", "who", "does", "not",
    "more", "some", "one", "two", "way", "get", "set", "need",
}

# 技术术语正则：匹配代码标识符（函数名、类名、API 调用链）
_TECH_TERM_RE = re.compile(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*")


class OutputGuard:
    """输出护栏：来源一致性校验 + 来源链接"""

    @staticmethod
    def extract_entities(text: str) -> set[str]:
        """提取回答中的关键实体

        中英文混合提取：
        - 英文：代码标识符（含调用链）匹配，过滤停用词
        - 中文：jieba 分词取非 ASCII 短语（长度≥2）
        """
        import jieba

        entities = set()

        # 英文代码标识符
        for match in _TECH_TERM_RE.finditer(text):
            token = match.group().strip().lower()
            if (
                len(token) >= 3
                and len(token) <= 50
                and token not in _STOP_WORDS
                and not token.isdigit()
                and not token.startswith("http")
            ):
                entities.add(token)

        # 中文分词
        for word in jieba.cut(text):
            word = word.strip()
            if len(word) >= 2 and not word.isascii():
                entities.add(word)

        return entities

    @staticmethod
    def verify_sources(
        response: str, retrieved_docs: list[dict]
    ) -> tuple[bool, str]:
        """检查回答中的关键实体是否在检索文档中出现

        Returns:
            (passed, warning_message)
        """
        if not retrieved_docs:
            return True, ""

        # 提取实体
        answer_entities = OutputGuard.extract_entities(response)

        # 汇总所有检索文档的文本
        doc_text = " ".join(
            d["text"].lower() for d in retrieved_docs
        )

        # 检查每个实体是否出现在文档中
        if not answer_entities:
            return True, ""

        found = 0
        for entity in answer_entities:
            if entity in doc_text:
                found += 1

        ratio = found / len(answer_entities)
        if ratio < 0.5:
            warning = (
                "Note: Some information in this answer could not be fully "
                "verified against the source documents. Please verify "
                "critical details independently."
            )
            return False, warning

        return True, ""

    @staticmethod
    def append_sources(response: str, retrieved_docs: list[dict]) -> str:
        """追加来源链接"""
        sources = set()
        for d in retrieved_docs:
            src = d["metadata"].get("source", "")
            if src and src != "unknown":
                sources.add(src)

        if not sources:
            return response

        source_lines = "\n".join(f"- {s}" for s in sorted(sources))
        return f"{response}\n\n---\nReferences:\n{source_lines}"
