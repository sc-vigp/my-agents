"""
Single Agent — core agent class.

The agent follows a simple ReAct-style loop:
  1. Send the conversation history (including any previous tool results) to
     the model.
  2. If the model requests one or more tool calls, execute them and append
     the results to the conversation.
  3. Repeat until the model produces a final text reply (no tool calls).
"""

import json
import os
from typing import Iterator

import openai

from .tools import TOOL_SCHEMAS, dispatch

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to a set of tools. "
    "Use the tools whenever they would help you give a more accurate or "
    "useful answer. Think step-by-step when solving problems."
)

DEFAULT_MODEL = "gpt-4o-mini"


class Agent:
    """A single AI agent backed by an OpenAI chat model with tool-use."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        api_key: str | None = None,
        max_tool_rounds: int = 10,
    ) -> None:
        """
        Initialise the agent.

        Parameters
        ----------
        model:
            OpenAI model identifier (default: ``gpt-4o-mini``).
        system_prompt:
            The system message sent at the start of every conversation.
        api_key:
            OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
            variable when *None*.
        max_tool_rounds:
            Safety cap on the number of tool-call/response cycles per turn.
        """
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key found. "
                "Set the OPENAI_API_KEY environment variable or pass api_key=."
            )

        self._client = openai.OpenAI(api_key=resolved_key)
        self._model = model
        self._max_tool_rounds = max_tool_rounds

        self._messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Send *user_message* to the agent and return its final text reply.

        Tool calls are handled transparently inside this method.
        """
        self._messages.append({"role": "user", "content": user_message})

        for _ in range(self._max_tool_rounds):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )

            choice = response.choices[0]
            assistant_msg = choice.message

            # Append the raw assistant message to history
            self._messages.append(assistant_msg.model_dump(exclude_unset=False))

            # If no tool calls, we have the final answer
            if not assistant_msg.tool_calls:
                return assistant_msg.content or ""

            # Execute each requested tool and record the results
            for tool_call in assistant_msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                result = dispatch(tool_name, tool_args)

                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        # Fallback if we exhausted the tool rounds
        return "Agent reached the maximum number of tool-call rounds without a final answer."

    def chat_stream(self, user_message: str) -> Iterator[str]:
        """
        Send *user_message* and yield the final reply token-by-token.

        Tool calls are executed silently before streaming begins.
        """
        # Use the non-streaming path to handle any tool rounds, then stream
        # the final response by replaying it with streaming enabled.

        self._messages.append({"role": "user", "content": user_message})

        for _ in range(self._max_tool_rounds):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )

            choice = response.choices[0]
            assistant_msg = choice.message
            self._messages.append(assistant_msg.model_dump(exclude_unset=False))

            if not assistant_msg.tool_calls:
                # Stream the final answer
                stream = self._client.chat.completions.create(
                    model=self._model,
                    messages=self._messages[:-1],  # exclude the prefetched reply
                    tools=TOOL_SCHEMAS,
                    tool_choice="none",
                    stream=True,
                )
                full_reply = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_reply += delta.content
                        yield delta.content
                # Update history with the streamed reply
                self._messages[-1] = {
                    "role": "assistant",
                    "content": full_reply,
                }
                return

            for tool_call in assistant_msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                result = dispatch(tool_name, tool_args)
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        yield "Agent reached the maximum number of tool-call rounds without a final answer."

    def reset(self) -> None:
        """Clear conversation history (preserves the system prompt)."""
        self._messages = [self._messages[0]]
