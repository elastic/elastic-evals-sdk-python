"""Kibana-backed inference client for elastic-evals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list["ToolCall"] | None = None


class ToolCall(BaseModel):
    id: str | None = None
    type: str = "function"
    function: dict[str, Any] | None = None


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: ChatMessage
    tool_calls: list[ToolCall] | None = None
    usage: dict[str, Any] | None = None

    @classmethod
    def from_kibana_response(cls, payload: dict[str, Any]) -> "ChatCompletionResponse":
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            message = ChatMessage(role="assistant", content=str(data))
            return cls(message=message)

        message_data: dict[str, Any] | None = None
        tool_calls: list[ToolCall] | None = None
        usage = data.get("usage")

        if data.get("choices"):
            choice = data["choices"][0]
            message_data = choice.get("message") if isinstance(choice, dict) else None
        elif "message" in data:
            if isinstance(data["message"], dict):
                message_data = data["message"]
            else:
                message_data = {"role": "assistant", "content": data["message"]}

        if message_data is None:
            message_data = {"role": "assistant", "content": None}

        message = ChatMessage(
            role=message_data.get("role", "assistant"),
            content=message_data.get("content"),
        )

        raw_tool_calls = message_data.get("tool_calls") or data.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            tool_calls = [ToolCall.model_validate(item) for item in raw_tool_calls]
            message.tool_calls = tool_calls

        return cls(message=message, tool_calls=tool_calls, usage=usage)


class PromptToolCall(BaseModel):
    toolCallId: str | None = None
    function: dict[str, Any]


class PromptResponse(BaseModel):
    content: str | None = None
    tool_calls: list[PromptToolCall] | None = None

    @classmethod
    def from_kibana_response(cls, payload: dict[str, Any]) -> "PromptResponse":
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            return cls(content=str(data))

        tool_calls = data.get("toolCalls") or data.get("tool_calls")
        parsed_tool_calls = None
        if isinstance(tool_calls, list):
            parsed_tool_calls = [PromptToolCall.model_validate(item) for item in tool_calls]

        return cls(content=data.get("content"), tool_calls=parsed_tool_calls)


@dataclass
class KibanaInferenceError(RuntimeError):
    message: str
    status_code: int | None = None
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


def _is_retryable_error(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in {408, 429} or error.response.status_code >= 500
    if isinstance(error, httpx.HTTPError):
        return True
    if isinstance(error, KibanaInferenceError):
        return error.retryable
    return False


class KibanaInferenceClient:
    def __init__(
        self,
        kibana_url: str,
        connector_id: str,
        auth: str,
        timeout: float = 120.0,
    ) -> None:
        self.kibana_url = kibana_url.rstrip("/")
        self.connector_id = connector_id
        self.auth = auth
        self.timeout = timeout

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable_error),
    )
    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        *,
        sub_action: str = "invokeAI",
        model: str | None = None,
        temperature: float | None = None,
    ) -> ChatCompletionResponse:
        url = f"{self.kibana_url}/api/actions/connector/{self.connector_id}/_execute"

        sub_action_params: dict[str, Any] = {"messages": messages}
        if tools is not None:
            sub_action_params["tools"] = tools
        if tool_choice is not None:
            sub_action_params["tool_choice"] = tool_choice
        if model is not None:
            sub_action_params["model"] = model
        if temperature is not None:
            sub_action_params["temperature"] = temperature

        payload = {
            "params": {
                "subAction": sub_action,
                "subActionParams": sub_action_params,
            }
        }

        headers = {
            "kbn-xsrf": "true",
            "Authorization": f"Basic {self.auth}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                message = f"Kibana inference request failed with {exc.response.status_code}"
                raise KibanaInferenceError(message, status_code=exc.response.status_code) from exc

            payload_data = response.json()

        if isinstance(payload_data, dict) and payload_data.get("status") == "error":
            message = payload_data.get("message") or "Kibana inference request failed"
            service_message = payload_data.get("service_message")
            retry_flag = payload_data.get("retry")
            retryable = bool(retry_flag)
            if service_message:
                message = f"{message} ({service_message})"
            raise KibanaInferenceError(message, retryable=retryable)

        return ChatCompletionResponse.from_kibana_response(payload_data)

    async def prompt(
        self,
        *,
        prompt: dict[str, Any],
        input_data: dict[str, Any],
        tool_choice: dict[str, Any] | None = None,
        temperature: float | None = None,
        model_name: str | None = None,
        connector_id: str | None = None,
    ) -> PromptResponse:
        url = f"{self.kibana_url}/internal/inference/prompt"
        payload: dict[str, Any] = {
            "connectorId": connector_id or self.connector_id,
            "prompt": prompt,
            "input": input_data,
        }
        if tool_choice is not None:
            payload["toolChoice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if model_name is not None:
            payload["modelName"] = model_name

        headers = {
            "kbn-xsrf": "true",
            "Authorization": f"Basic {self.auth}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                message = f"Kibana inference prompt request failed with {exc.response.status_code}"
                raise KibanaInferenceError(message, status_code=exc.response.status_code) from exc
            payload_data = response.json()

        if isinstance(payload_data, dict) and payload_data.get("type") == "error":
            message = payload_data.get("message") or "Kibana inference prompt request failed"
            raise KibanaInferenceError(message)

        return PromptResponse.from_kibana_response(payload_data)
