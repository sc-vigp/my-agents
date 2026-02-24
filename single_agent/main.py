"""
Interactive CLI for the single agent system.

Run:
    python -m single_agent.main

or:
    python single_agent/main.py

Environment variables:
    OPENAI_API_KEY  – your OpenAI API key (required)
    AGENT_MODEL     – model to use (default: gpt-4o-mini)
"""

import os
import sys

from .agent import Agent, DEFAULT_MODEL

BANNER = """\
╔══════════════════════════════════════════════════════╗
║           Single Agent System — Interactive CLI       ║
╠══════════════════════════════════════════════════════╣
║  Available tools:                                     ║
║    • calculator      – evaluate math expressions      ║
║    • get_current_datetime – return current date/time  ║
║    • count_words     – count words in text            ║
║    • reverse_text    – reverse a string               ║
╠══════════════════════════════════════════════════════╣
║  Commands:                                            ║
║    /reset   – clear conversation history              ║
║    /quit    – exit the programme                      ║
╚══════════════════════════════════════════════════════╝
"""


def _check_api_key() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "\n[ERROR] OPENAI_API_KEY is not set.\n"
            "Export it before running:\n"
            "  export OPENAI_API_KEY='sk-...'\n",
            file=sys.stderr,
        )
        return False
    return True


def run_cli() -> None:
    if not _check_api_key():
        sys.exit(1)

    model = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
    agent = Agent(model=model)

    print(BANNER)
    print(f"  Model: {model}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/reset":
            agent.reset()
            print("[Conversation history cleared]\n")
            continue

        print("Agent: ", end="", flush=True)
        try:
            for token in agent.chat_stream(user_input):
                print(token, end="", flush=True)
            print()  # newline after the full reply
        except Exception as exc:  # noqa: BLE001
            print(f"\n[Error] {exc}", file=sys.stderr)

        print()


if __name__ == "__main__":
    run_cli()
