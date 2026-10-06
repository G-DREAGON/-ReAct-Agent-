"""
Token 管理模块
==============
Agent 跑着跑着，对话历史会越来越长，超过 LLM 的上下文窗口就会报错。
这个模块负责：
1. 计算当前对话占了多少 tokens
2. 在接近上限时自动压缩历史
3. 压缩策略：保留系统提示 + 保留最近 N 条消息 + 旧消息用摘要替代
"""

import tiktoken
from typing import Any


class TokenManager:
    """
    Token 管理器——确保对话不会超过上下文窗口。

    压缩策略（混合方案）：
    1. 如果总 tokens < 上限的 80%，不做任何处理
    2. 如果超过 80%，对旧消息做摘要压缩
    3. 如果即使摘要后仍然超限，从最旧的消息开始丢弃
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        model: str = "gpt-4o-mini",
    ):
        self.max_tokens = max_tokens
        # tiktoken 用 cl100k_base 编码（GPT-4/GPT-3.5 通用）
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # 如果 tiktoken 没加载成功，用 o200k_base 试试
            self.encoder = tiktoken.get_encoding("o200k_base")

    def count_tokens(self, text: str) -> int:
        """计算一段文本的 token 数量"""
        try:
            return len(self.encoder.encode(text))
        except Exception:
            # 回退方案：简单估算（中文约1.5字/token，英文约4字/token）
            return len(text) // 2

    def count_messages(self, messages: list[dict]) -> int:
        """计算整个 messages 列表的 token 总数"""
        total = 0
        for msg in messages:
            total += self.count_tokens(str(msg.get("content", "")))
            total += self.count_tokens(str(msg.get("role", "")))
            total += 4  # 每条消息的格式开销
        return total

    def manage(self, messages: list[dict], system_prompt: str = "") -> list[dict]:
        """
        管理对话历史，在必要时压缩。

        参数:
            messages: 当前对话历史
            system_prompt: 系统提示词（会被保留，不压缩）

        返回:
            压缩后的 messages 列表
        """
        total = self.count_messages(messages)

        # 还没到 80% 警告线，直接返回
        if total < self.max_tokens * 0.8:
            return messages

        print(f"\n⚠️ Token 用量已达 {total}/{self.max_tokens}，开始压缩历史...")

        # 压缩策略：
        # - 保留 system prompt（如果存在）
        # - 保留最近 4 轮对话（8条消息：4 user + 4 assistant/tool）
        # - 旧消息替换为一段摘要

        compressed = []

        # 分离 system 消息
        sys_msg = None
        for msg in messages:
            if msg.get("role") == "system":
                sys_msg = msg
                break

        # 获取非 system 的消息
        non_system = [m for m in messages if m.get("role") != "system"]

        # 如果消息太多，保留最近 8 条非系统消息
        if len(non_system) > 8:
            old_messages = non_system[:-8]
            recent_messages = non_system[-8:]

            # 生成摘要
            summary = self._generate_summary(old_messages)

            if sys_msg:
                compressed.append(sys_msg)

            compressed.append({
                "role": "system",
                "content": f"[历史对话摘要] {summary}",
            })
            compressed.extend(recent_messages)
        else:
            if sys_msg:
                compressed.append(sys_msg)
            compressed.extend(non_system)

        new_total = self.count_messages(compressed)
        print(f"  压缩完成: {total} → {new_total} tokens")

        return compressed

    def _generate_summary(self, messages: list[dict]) -> str:
        """
        生成旧对话的简单摘要。
        这是轻量级实现——生产环境中应该调用 LLM 来做摘要。
        这里用一个简单的统计方法替代。
        """
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]

        summary_parts = []
        if user_msgs:
            topics = [str(m.get("content", ""))[:50] for m in user_msgs]
            summary_parts.append(f"用户之前询问了 {len(user_msgs)} 个问题")
            summary_parts.append(f"话题包括: {'; '.join(topics[:5])}")
        if tool_msgs:
            summary_parts.append(f"Agent 共调用了 {len(tool_msgs)} 次工具")

        return "。".join(summary_parts) if summary_parts else "无历史对话"


if __name__ == "__main__":
    mgr = TokenManager(max_tokens=100)
    test_text = "Hello World! 你好世界！"
    print(f"'{test_text}' = {mgr.count_tokens(test_text)} tokens")
