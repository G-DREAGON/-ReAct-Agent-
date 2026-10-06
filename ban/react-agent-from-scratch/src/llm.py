"""
LLM 调用封装层
==============
职责：封装对 OpenAI 兼容 API 的调用，包括：
- 普通对话（同步）
- 流式输出（边生成边返回）
- 支持 Function Calling / Tool Use 模式
- 支持任意兼容 OpenAI 协议的 API（DeepSeek、通义千问等）
"""

import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ToolCall:
    """LLM 返回的工具调用请求"""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """LLM 的一次回复"""
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient:
    """OpenAI 兼容 API 的轻量封装"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if not self.api_key or self.api_key == "sk-your-key-here":
            raise ValueError(
                "请先设置 OPENAI_API_KEY！\n"
                "方式一：在 .env 文件中填入你的 API Key\n"
                "方式二：设置环境变量 OPENAI_API_KEY=你的key"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        """发送消息给 LLM，返回回复。

        Args:
            messages: 对话历史 [{"role": "system/user/assistant/tool", "content": "..."}]
            tools:    工具定义列表（Function Calling 的 functions schema）
            stream:   是否启用流式输出

        Returns:
            LLMResponse 对象，包含文本内容或工具调用列表
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        return self._parse_response(choice)

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ):
        """流式对话生成器，逐 token yield"""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = self.client.chat.completions.create(**kwargs)

        collected_content = ""
        collected_tool_calls: list[dict] = []

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            if delta.content:
                collected_content += delta.content
                yield {"type": "text", "content": delta.content}

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    while len(collected_tool_calls) <= idx:
                        collected_tool_calls.append({"id": "", "name": "", "arguments": ""})
                    if tc_delta.id:
                        collected_tool_calls[idx]["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        collected_tool_calls[idx]["name"] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        collected_tool_calls[idx]["arguments"] += tc_delta.function.arguments

        yield {
            "type": "done",
            "content": collected_content,
            "tool_calls": [
                ToolCall(id=tc["id"], name=tc["name"], arguments=self._safe_json_parse(tc["arguments"]))
                for tc in collected_tool_calls
                if tc["name"]
            ],
        }

    def _parse_response(self, choice) -> LLMResponse:
        """把 OpenAI SDK 的 Choice 对象转成 LLMResponse"""
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=self._safe_json_parse(tc.function.arguments),
                    )
                )

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
        )

    @staticmethod
    def _safe_json_parse(s: str) -> dict[str, Any]:
        """安全解析 JSON，失败返回空字典"""
        import json
        try:
            return json.loads(s) if isinstance(s, str) else (s or {})
        except json.JSONDecodeError:
            return {"_raw": s}


if __name__ == "__main__":
    try:
        client = LLMClient()
        resp = client.chat([{"role": "user", "content": "你好，用一句话介绍你自己"}])
        print(f"LLM OK: {resp.content}")
    except ValueError as e:
        print(f"配置错误: {e}")
    except Exception as e:
        print(f"调用失败: {e}")
