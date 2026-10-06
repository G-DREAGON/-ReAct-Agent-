"""
ReAct Agent 核心引擎
====================
这是整个项目的灵魂——实现 Thought → Action → Observation 循环。

设计思想：
1. 使用 OpenAI 原生 Function Calling（而非文本解析），更稳定可靠
2. 同时保留纯文本 ReAct 推理链的可读性
3. 支持流式输出，让用户看到 Agent"边想边做"
4. 内置重试、步数限制、错误降级等工程保护

核心循环（伪代码）：
    while 还没得到最终答案 and 步数未超限:
        1. 检查 token 用量，超了就压缩历史
        2. 发 messages + tools schema 给 LLM
        3. 如果 LLM 返回工具调用 → 执行工具 → 结果加入对话 → 回到步骤2
        4. 如果 LLM 返回纯文本 → 这就是最终答案，结束循环
"""

import json
import os
import time
from typing import Any

from dotenv import load_dotenv

from .llm import LLMClient, LLMResponse
from .tools import ToolRegistry, create_default_registry
from .token_manager import TokenManager

load_dotenv()

# ---------------------------------------------------------------------------
# System Prompt: 告诉 LLM 它是什么角色、怎么使用工具
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一个具备自主规划和工具使用能力的 AI Agent。

## 你的能力
你可以使用以下工具来完成任务：
- get_current_time: 获取当前时间
- calculator: 执行数学计算
- search_web: 搜索互联网信息
- read_file: 读取文件
- write_file: 写入文件

## 工作原则
1. 收到用户请求后，先思考需要什么信息，再决定调用哪些工具
2. 每次只调用必要的工具，不要多余调用
3. 工具返回结果后，分析结果再决定下一步
4. 当你有足够信息回答用户时，直接给出清晰、完整的回答
5. 回答时尽量使用中文

## 注意事项
- 如果你不确定某个信息，使用 search_web 搜索而不是猜测
- 数学计算务必使用 calculator 工具，不要心算
- 回复简洁有力，不要啰嗦
"""


class ReActAgent:
    """
    ReAct Agent —— 从零实现的自主推理型智能体。

    用法:
        agent = ReActAgent()
        agent.run("帮我算一下 123 * 456 等于多少，然后搜索一下 Agent 开发的最新趋势")
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: ToolRegistry | None = None,
        token_manager: TokenManager | None = None,
        max_steps: int | None = None,
        verbose: bool = True,
    ):
        # LLM 客户端
        self.llm = llm or LLMClient()

        # 工具注册中心
        self.tools = tools or create_default_registry()

        # Token 管理器
        max_ctx = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))
        self.token_manager = token_manager or TokenManager(max_tokens=max_ctx)

        # 最大执行步数（防止无限循环）
        self.max_steps = max_steps or int(os.getenv("MAX_STEPS", "10"))

        # 是否打印详细日志
        self.verbose = verbose

        # 运行统计
        self.stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 主入口：运行 Agent
    # ------------------------------------------------------------------
    def run(self, user_input: str) -> str:
        """
        运行 Agent，处理用户输入，返回最终回答。

        这是面向使用者的主接口。
        """
        self.stats = {"start_time": time.time(), "steps": 0, "tool_calls": []}

        # 初始化对话
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        # ==============================================================
        # 核心循环：Thought → Action → Observation
        # ==============================================================
        while self.stats["steps"] < self.max_steps:
            self.stats["steps"] += 1
            step = self.stats["steps"]

            if self.verbose:
                print(f"\n{'─'*50}")
                print(f"🔄 第 {step}/{self.max_steps} 步")
                print(f"{'─'*50}")

            # ---- 第1层保护：Token 管理 ----
            messages = self.token_manager.manage(messages, SYSTEM_PROMPT)

            # ---- 第2层保护：获取 LLM 决策 ----
            try:
                response = self.llm.chat(
                    messages=messages,
                    tools=self.tools.get_schemas(),
                )
            except Exception as e:
                if self.verbose:
                    print(f"❌ LLM 调用失败: {e}")
                return f"抱歉，调用 LLM 时出错: {e}"

            # ---- 没有工具调用 → 这就是最终答案 ----
            if not response.has_tool_calls():
                final_answer = response.content or ""
                self.stats["end_time"] = time.time()
                self.stats["final_answer"] = final_answer

                if self.verbose:
                    print(f"\n✅ Agent 完成任务（共 {step} 步，耗时 {self.stats['end_time'] - self.stats['start_time']:.1f}s）")

                return final_answer

            # ---- 有工具调用 → 执行工具然后继续循环 ----
            # 先把 LLM 的回复加入对话
            assistant_msg = self._build_assistant_message(response)
            messages.append(assistant_msg)

            # 逐个执行工具调用
            for tool_call in response.tool_calls:
                if self.verbose:
                    print(f"🔧 调用工具: {tool_call.name}({json.dumps(tool_call.arguments, ensure_ascii=False)})")

                tool_result = self.tools.execute(tool_call.name, tool_call.arguments)

                if self.verbose:
                    # 截断过长的结果
                    display = tool_result[:200] + ("..." if len(tool_result) > 200 else "")
                    print(f"📊 工具返回: {display}")

                # 记录统计
                self.stats["tool_calls"].append({
                    "step": step,
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                    "result": tool_result[:500],
                })

                # 把工具结果加入对话
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

        # ---- 超过最大步数 ----
        if self.verbose:
            print(f"\n⚠️ 达到最大步数限制 ({self.max_steps})，强制终止")

        self.stats["end_time"] = time.time()
        return f"抱歉，我尝试了 {self.max_steps} 步仍未完成任务。请简化你的问题。最后一步的思考：{response.content}"

    # ------------------------------------------------------------------
    # 流式运行（实时展示思考过程）
    # ------------------------------------------------------------------
    def run_stream(self, user_input: str):
        """
        流式运行 Agent，逐 token 输出，让用户看到 Agent"边想边做"。

        yield 的格式:
            {"type": "thinking", "content": "..."}   - 思考过程
            {"type": "tool_call", "name": "...", "args": {...}}  - 工具调用
            {"type": "tool_result", "content": "..."}  - 工具返回
            {"type": "answer", "content": "..."}   - 最终回答
            {"type": "error", "content": "..."}    - 错误
        """
        self.stats = {"start_time": time.time(), "steps": 0, "tool_calls": []}

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        while self.stats["steps"] < self.max_steps:
            self.stats["steps"] += 1
            step = self.stats["steps"]

            yield {"type": "step", "content": f"第 {step}/{self.max_steps} 步"}

            messages = self.token_manager.manage(messages, SYSTEM_PROMPT)

            # 流式调用 LLM
            try:
                stream_gen = self.llm.chat_stream(
                    messages=messages,
                    tools=self.tools.get_schemas(),
                )
            except Exception as e:
                yield {"type": "error", "content": str(e)}
                return

            response = None
            for chunk in stream_gen:
                if chunk["type"] == "text":
                    yield {"type": "thinking", "content": chunk["content"]}
                elif chunk["type"] == "done":
                    response = LLMResponse(
                        content=chunk.get("content", ""),
                        tool_calls=chunk.get("tool_calls", []),
                    )

            if response is None:
                yield {"type": "error", "content": "LLM 返回为空"}
                return

            # 没有工具调用 → 最终答案
            if not response.has_tool_calls():
                self.stats["end_time"] = time.time()
                yield {"type": "answer", "content": response.content or ""}
                return

            # 有工具调用
            assistant_msg = self._build_assistant_message(response)
            messages.append(assistant_msg)

            for tool_call in response.tool_calls:
                yield {"type": "tool_call", "name": tool_call.name, "args": tool_call.arguments}

                tool_result = self.tools.execute(tool_call.name, tool_call.arguments)
                yield {"type": "tool_result", "content": tool_result[:200]}

                self.stats["tool_calls"].append({
                    "step": step,
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

        self.stats["end_time"] = time.time()
        yield {"type": "error", "content": f"达到最大步数限制 ({self.max_steps})"}

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _build_assistant_message(self, response: LLMResponse) -> dict:
        """把 LLMResponse 转成标准的 assistant message"""
        msg: dict = {"role": "assistant", "content": response.content or None}

        if response.has_tool_calls():
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in response.tool_calls
            ]

        return msg

    def get_stats(self) -> dict:
        """获取运行统计信息"""
        elapsed = self.stats.get("end_time", time.time()) - self.stats.get("start_time", time.time())
        return {
            **self.stats,
            "elapsed_seconds": round(elapsed, 2),
            "tool_call_count": len(self.stats.get("tool_calls", [])),
        }


# ======================================================================
# 快速测试
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  ReAct Agent - 从零实现演示")
    print("=" * 60)

    agent = ReActAgent()

    # 测试1: 单工具调用
    print("\n\n📝 测试1: 简单计算 + 搜索")
    result = agent.run("帮我算一下 256 * 128 等于多少，然后搜索一下今天北京的天气")
    print(f"\n🎯 最终回答:\n{result}")

    # 测试2: 需要多步推理
    print("\n\n📝 测试2: 多步推理")
    result = agent.run("现在几点了？如果现在距离明天上午9点还有多久？")
    print(f"\n🎯 最终回答:\n{result}")
