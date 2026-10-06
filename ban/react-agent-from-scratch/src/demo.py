"""
ReAct Agent 演示脚本
====================
运行方式: python -m src.demo

包含多个场景演示，展示 Agent 的不同能力：
1. 基础推理（不需要工具）
2. 单工具调用（计算）
3. 多工具链式调用（搜索+计算+写文件）
4. 流式输出演示
5. 错误处理
"""

import sys
import os

# 确保可以找到 src 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import ReActAgent
from src.llm import LLMClient


def print_separator(title: str):
    print(f"\n\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def demo_basic_chat(agent: ReActAgent):
    """演示1: 不需要工具的基础对话"""
    print_separator("演示1: 基础对话（不需要工具）")
    result = agent.run("你好，请用三句话介绍一下什么是 AI Agent")
    print(f"\n🎯 Agent 回答:\n{result}")


def demo_single_tool(agent: ReActAgent):
    """演示2: 需要调用单个工具"""
    print_separator("演示2: 单工具调用 - 数学计算")
    result = agent.run("我是一个大学生，帮我算一下：如果每月定投2000元，年化收益率8%，复利计算20年后总共有多少钱？")
    print(f"\n🎯 Agent 回答:\n{result}")


def demo_multi_tool_chain(agent: ReActAgent):
    """演示3: 多工具链式调用"""
    print_separator("演示3: 多工具链式调用 - 综合任务")
    result = agent.run(
        "请帮我做以下事情：\n"
        "1. 查一下现在的时间\n"
        "2. 搜索一下'2026年AI Agent开发'的最新动态\n"
        "3. 把以上信息整理成一份摘要，写入 agent_report.txt 文件"
    )
    print(f"\n🎯 Agent 回答:\n{result}")


def demo_stream(agent: ReActAgent):
    """演示4: 流式输出"""
    print_separator("演示4: 流式输出（实时看到Agent思考过程）")
    print("Agent 思考中...\n")

    for chunk in agent.run_stream("先查时间，然后算一下 1024 * 2048 等于多少"):
        if chunk["type"] == "thinking":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "tool_call":
            print(f"\n🔧 [{chunk['name']}] ", end="", flush=True)
        elif chunk["type"] == "tool_result":
            print(f"→ {chunk['content'][:100]}", flush=True)
        elif chunk["type"] == "answer":
            print(f"\n\n✅ 最终回答: {chunk['content']}")
        elif chunk["type"] == "error":
            print(f"\n❌ 错误: {chunk['content']}")


def main():
    print("=" * 70)
    print("  🧠 ReAct Agent — 从零实现演示")
    print("=" * 70)

    # 初始化 Agent
    try:
        agent = ReActAgent(verbose=True)
    except ValueError as e:
        print(f"\n❌ 初始化失败: {e}")
        print("\n请先配置 .env 文件:")
        print("  1. 复制 .env.example 为 .env")
        print("  2. 填入你的 OPENAI_API_KEY")
        print("  3. 如果使用 DeepSeek，修改 OPENAI_BASE_URL 和 OPENAI_MODEL")
        return

    print(f"✅ Agent 初始化成功")
    print(f"   LLM 模型: {agent.llm.model}")
    print(f"   可用工具: {agent.tools.list_tools()}")
    print(f"   最大步数: {agent.max_steps}")

    # 运行演示
    try:
        demo_basic_chat(agent)
    except Exception as e:
        print(f"演示1 失败: {e}")

    try:
        demo_single_tool(agent)
    except Exception as e:
        print(f"演示2 失败: {e}")

    try:
        demo_multi_tool_chain(agent)
    except Exception as e:
        print(f"演示3 失败: {e}")

    try:
        demo_stream(agent)
    except Exception as e:
        print(f"演示4 失败: {e}")

    # 打印统计
    print_separator("运行统计")
    stats = agent.get_stats()
    print(f"  总步数: {stats.get('steps')}")
    print(f"  工具调用次数: {stats.get('tool_call_count')}")
    print(f"  耗时: {stats.get('elapsed_seconds')}s")


if __name__ == "__main__":
    main()
