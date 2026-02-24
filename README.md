# my-agents

This repo is my very first agent development from scratch. It contains my exploration of Agentic AI, including Single Agent and Multi Agent Systems.

---

## Single Agent System

A minimal but complete **ReAct-style** single agent that:

- Accepts natural-language input from you in an interactive CLI loop.
- Decides when to call built-in **tools** to answer your question more accurately.
- Streams the final reply token-by-token so you see output in real time.
- Maintains **conversation history** across turns, so it understands context.

### Built-in tools

| Tool | What it does |
|------|-------------|
| `calculator` | Evaluates math expressions (`2 + 3 * 4`, `sqrt(144)`, …) |
| `get_current_datetime` | Returns the current local date and time |
| `count_words` | Counts the words in a piece of text |
| `reverse_text` | Reverses a string character-by-character |

### Prerequisites

- Python ≥ 3.12
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/sc-vigp/my-agents.git
cd my-agents

# 2. Install dependencies
pip install -r requirements.txt

# 3. Export your OpenAI API key
export OPENAI_API_KEY="sk-..."

# 4. (Optional) choose a different model (default: gpt-4o-mini)
export AGENT_MODEL="gpt-4o"
```

### Run the interactive CLI

```bash
python -m single_agent.main
```

You will see a banner and a `You:` prompt. Type any message and press **Enter**.

```
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

  Model: gpt-4o-mini

You: What is 123 * 456?
Agent: 123 multiplied by 456 equals 56,088.

You: What time is it?
Agent: The current date and time is 2026-02-24 07:25:26.

You: /quit
Goodbye!
```

### Use the agent from Python

```python
from single_agent.agent import Agent

agent = Agent()  # reads OPENAI_API_KEY from environment

reply = agent.chat("What is 2 to the power of 10?")
print(reply)  # "2 to the power of 10 is 1024."

# Streaming
for token in agent.chat_stream("Reverse the word 'python'"):
    print(token, end="", flush=True)

# Reset conversation history
agent.reset()
```

### Run the tests

```bash
python -m pytest tests/ -v
```

### Project structure

```
my-agents/
├── requirements.txt          # Python dependencies
├── single_agent/
│   ├── __init__.py
│   ├── agent.py              # Agent class (ReAct loop + streaming)
│   ├── tools.py              # Tool implementations + OpenAI schemas
│   └── main.py               # Interactive CLI entry point
└── tests/
    └── test_tools.py         # Unit tests for all built-in tools
```
