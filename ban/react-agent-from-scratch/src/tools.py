"""
工具系统（Tool System）
======================
Agent 的"双手"——定义工具、注册工具、执行工具。

核心概念：
1. 每个工具 = 名称 + 描述 + 参数schema + 执行函数
2. ToolRegistry 统一管理所有工具
3. 工具的 schema 会被发给 LLM，LLM 据此决定调哪个工具、传什么参数

类比理解：
- 就像你手机上的 App，每个 App 有自己的界面（schema）和功能（execute）
- Agent 就是那个"手"，根据需求打开对应的 App 并输入参数
"""

import json
import math
from datetime import datetime
from typing import Any, Callable


class Tool:
    """
    一个工具 = 一段可以被 Agent 调用的函数。

    属性：
        name:        工具名称，LLM 用这个名字来调用
        description: 工具的功能描述，LLM 据此判断"什么时候该用这个工具"
        parameters:  参数的 JSON Schema，LLM 据此知道"该传什么参数"
        func:        实际执行的 Python 函数
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., str],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def to_openai_schema(self) -> dict:
        """
        转为 OpenAI Function Calling 需要的格式。
        这就是发给 LLM 的"工具说明书"。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        """执行工具，返回字符串结果"""
        try:
            result = self.func(**arguments)
            return str(result)
        except Exception as e:
            return f"工具执行失败: {e}"


class ToolRegistry:
    """工具注册中心——管理所有可用工具"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """按名称获取工具"""
        return self._tools.get(name)

    def get_schemas(self) -> list[dict]:
        """获取所有工具的 OpenAI schema 列表，用于发给 LLM"""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """按名称执行工具"""
        tool = self.get(name)
        if tool is None:
            return f"错误: 未知工具 '{name}'，可用工具: {list(self._tools.keys())}"
        return tool.execute(arguments)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


# ======================================================================
# 内置工具：一组实用的工具函数
# ======================================================================

def _get_current_time() -> str:
    """获取当前日期和时间"""
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}，星期{['一','二','三','四','五','六','日'][now.weekday()]}"


def _calculator(expression: str) -> str:
    """
    安全计算器——只能做数学运算，不会执行恶意代码。
    支持: + - * / ** sqrt() sin() cos() abs() 等
    """
    # 只允许数学相关的函数和运算符
    allowed_names = {
        "abs": abs, "round": round, "max": max, "min": min,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10,
        "pi": math.pi, "e": math.e, "pow": pow, "ceil": math.ceil,
        "floor": math.floor, "int": int, "float": float,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def _search_web(query: str) -> str:
    """
    模拟网络搜索——实际项目中可以接入真实搜索 API（如 SerpAPI、Bing Search）。
    这里返回模拟数据用于演示 Agent 的工具调用流程。
    """
    mock_db = {
        "北京天气": "北京今天晴，15°C~25°C，北风3级，空气质量良。明天多云转阴，12°C~20°C。",
        "python最新版本": "Python 最新稳定版是 3.14.0（2025年10月发布）。主要新特性：模式匹配增强、更快的解析器、改进的GIL。",
        "agent开发": "AI Agent 开发的核心技术栈：LangChain/LangGraph（编排）、MCP协议（工具集成）、RAG（检索增强）、ReAct/Plan-and-Solve（推理范式）。",
    }
    for key, value in mock_db.items():
        if key in query:
            return value
    return f'关于"{query}"的搜索结果：这是一个模拟结果。在生产环境中，这里会接入真实的搜索API。'


def _read_file(filepath: str) -> str:
    """读取本地文件内容"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # 限制返回长度，避免撑爆上下文
        if len(content) > 2000:
            content = content[:2000] + f"\n... (文件共 {len(content)} 字符，已截断前2000字符)"
        return f"文件内容 ({filepath}):\n{content}"
    except FileNotFoundError:
        return f"错误: 文件不存在 '{filepath}'"
    except Exception as e:
        return f"读取文件失败: {e}"


def _write_file(filepath: str, content: str) -> str:
    """写入内容到本地文件"""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已成功写入文件: {filepath}（{len(content)} 字符）"
    except Exception as e:
        return f"写入文件失败: {e}"


# ======================================================================
# 工厂函数：创建带所有内置工具的 ToolRegistry
# ======================================================================

def create_default_registry() -> ToolRegistry:
    """创建一个预装了5个常用工具的 ToolRegistry"""
    registry = ToolRegistry()

    registry.register(Tool(
        name="get_current_time",
        description="获取当前的日期和时间。当你需要知道现在是什么时间时使用。",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        func=_get_current_time,
    ))

    registry.register(Tool(
        name="calculator",
        description="执行数学计算。支持四则运算、幂运算、三角函数等。传入数学表达式字符串。",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，例如 '2 + 3 * 4' 或 'sqrt(16)'",
                }
            },
            "required": ["expression"],
        },
        func=_calculator,
    ))

    registry.register(Tool(
        name="search_web",
        description="搜索互联网获取信息。当需要查找实时信息或不了解的内容时使用。",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                }
            },
            "required": ["query"],
        },
        func=_search_web,
    ))

    registry.register(Tool(
        name="read_file",
        description="读取本地文件的内容。参数为文件的完整路径。",
        parameters={
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "文件的完整路径",
                }
            },
            "required": ["filepath"],
        },
        func=_read_file,
    ))

    registry.register(Tool(
        name="write_file",
        description="将内容写入本地文件。参数为文件路径和要写入的内容。",
        parameters={
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "要写入的文件路径",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的内容",
                },
            },
            "required": ["filepath", "content"],
        },
        func=_write_file,
    ))

    return registry


if __name__ == "__main__":
    reg = create_default_registry()
    print(f"已注册工具: {reg.list_tools()}")
    print(f"\nSchemas (发给LLM的工具说明书):")
    print(json.dumps(reg.get_schemas(), ensure_ascii=False, indent=2))
