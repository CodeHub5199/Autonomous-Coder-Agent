# 🤖 Autonomous Coder Agent

> A production-grade, multi-agent autonomous coding system built with **LangGraph** and **LangChain**. Given a plain-English task, the system plans, writes, and executes code end-to-end — with a feedback loop for failure recovery and human-in-the-loop approval for sensitive operations.
---

## What It Does

You type: `"Build a FastAPI app with a /health endpoint and a Dockerfile"`

The agent:
1. **Plans** a multi-step execution strategy (directory setup → file creation → dependency install → run)
2. **Writes** each file with production-quality code
3. **Executes** shell commands step by step, streaming output live
4. **Repairs** broken code automatically using LLM-assisted error analysis
5. **Escalates** sensitive commands (like `curl`, `git push`, `sudo`) to you for approval before running

No manual scaffolding. No copy-pasting. Just results.

---

## System Architecture

```
_start_
   │
   ▼
planner  ──── generates a structured, step-by-step ExecutionPlan
   │
   ▼
router   ──── dispatches each step: write_file → writer | run_command → executor or HITL
   │
   ├──► writer   ──── generates full files or applies minimal diffs (incremental patching)
   │
   ├──► HITL     ──── prompts human approval for sensitive/destructive commands
   │         │
   │         └──► executor
   │
   └──► executor ──── runs shell commands; on failure:
              │
              ├──► repair  ──── LLM re-generates the broken file and re-runs
              │
              └──► (max retries exceeded) ──── human escalation → skip or abort
```

> See [`architecture.png`](./architecture.png) for the LangGraph state graph diagram.

### State Management

All agents share a single `AgentState` TypedDict that flows through the graph:

| Field | Purpose |
|---|---|
| `idea` | Original user task |
| `plan` | Structured `ExecutionPlan` with ordered `PlanStep` objects |
| `generated_code` | Append-only list of `{filename: content}` dicts (LangGraph reducer) |
| `current_step_idx` | Pointer to the active plan step |
| `retry_count` | Failure counter; triggers repair or human escalation |
| `hitl_approved` | Boolean result of the last HITL gate; `None` when not applicable |
| `execution_history` | Append-only audit trail of every event (LangGraph reducer) |
| `is_update_request` | Signals incremental edit mode vs. full project creation |

### Routing Logic

A central `router` function reads `current_step_idx` and the current step's `tool_name` to determine the next node. Post-execution routing handles three outcomes:

- **Success** → advance to next step via router
- **Failure + retries remaining** → `repair` (for file steps) or direct re-execution (for commands)
- **Failure + retries exhausted** → human escalation prompt; skip or abort

---

## Key Features

### 🧠 Three-Agent Pipeline
- **Planner** — Uses a structured-output LLM to decompose any task into an ordered list of `PlanStep` objects with explicit tool names, file targets, and expected outcomes.
- **Writer** — Generates complete, production-ready files. Detects whether a step requires a *full write* or a *minimal diff patch*, applying only the necessary change to avoid regressions.
- **Executor** — Runs shell commands with a 120s timeout, captures `stdout`/`stderr`, and streams both to the terminal in real time.

### 🔁 Feedback Loop & Auto-Repair
When a command fails, the system does not stop. It routes to the **Repair** agent, which receives the broken file content and the error message, generates a corrected version, writes it to disk, and re-runs the executor. Up to `MAX_RETRIES` (default: 3) attempts are made before escalating to the human.

### 🛡️ Human-in-the-Loop (HITL)
Before executing any command matching a curated list of sensitive patterns — `rm`, `chmod`, `sudo`, `curl`, `git push`, `npm publish`, `eval`, `kill`, and others — the agent pauses and presents a structured approval prompt. The decision is logged to the audit trail regardless of outcome. Certain commands (`rm -rf /`, `shutdown`, `reboot`, `mkfs`) are hard-blocked and never reach the HITL gate.

### ✂️ Incremental Patching
For update requests ("change the button color", "fix the import"), the Writer issues a minimal `search_block → replace_block` patch rather than regenerating the entire file. A fuzzy `SequenceMatcher` fallback handles cases where the LLM's search block doesn't match verbatim. The terminal renders a color-coded unified diff of every change.

### 📡 Real-Time Streaming
Agent reasoning and execution output are streamed word-by-word to the terminal using Rich's `Live` context manager. The LangGraph graph is invoked in `stream_mode="updates"` so each node's state patch is visible as it arrives.

### 🖥️ Rich Terminal UX
Color-coded agent headers, syntax-highlighted code previews (Monokai), spinners with elapsed-time counters, and a final summary table make the execution trace easy to follow. Each agent has its own color identity (cyan: Planner, green: Writer, yellow: Executor, magenta: HITL, red: Repair).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM integration | [LangChain](https://github.com/langchain-ai/langchain), [langchain-groq](https://github.com/langchain-ai/langchain-groq) |
| LLM provider | [Groq](https://groq.com/) (configurable via `GROQ_MODEL` env var) |
| Structured outputs | [Pydantic v2](https://docs.pydantic.dev/) with `.with_structured_output()` |
| Terminal UI | [Rich](https://github.com/Textualize/rich) |
| CLI | [Click](https://click.palletsprojects.com/) |
| Runtime | Python 3.11+ |

---

## Setup

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/autonomous-coder-agent.git
cd autonomous-coder-agent

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install langgraph langchain langchain-groq pydantic python-dotenv rich click
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b   # or any Groq-supported model
```

---

## Usage

### Single task (CLI)

```bash
# Pass the task as an argument
python final_v2.py run "Build a Python CLI todo app with add, list, and delete commands"

# Or get prompted interactively
python final_v2.py run
```

### Interactive REPL (multi-task session)

```bash
python final_v2.py interactive
```

### Options

```
python final_v2.py run --help

Options:
  --stream / --no-stream   Stream agent output in real time  [default: stream]
  --max-retries INTEGER    Max executor retries per step     [default: 3]
```

---

## Project Structure

```
.
├── final_v2.py          # Complete system — all agents, graph, and CLI
├── .env                 # API keys and model config (not committed)
├── architecture.png     # LangGraph state graph diagram
└── README.md
```

> The system is intentionally contained in a single file to make the architecture easy to read end-to-end. In a production deployment, each agent, the state schema, routing logic, and CLI would be split into separate modules.

---

## Sample Session

```
╔══════════════════════════════════════════════════════════════╗
║          🤖  Autonomous Coder Agent  v2.0                    ║
║          Planner → Writer → Executor  (LangGraph)            ║
╚══════════════════════════════════════════════════════════════╝
✓ Feedback loop (auto-retry)   ✓ Human-in-the-loop (HITL)
✓ Streaming output             ✓ Incremental patching

💡 What would you like to build or change? > Build a Flask REST API with a /users endpoint backed by SQLite

Task: Build a Flask REST API with a /users endpoint backed by SQLite

──────────────────────── Planner ─────────────────────────────
╭──────────────────────────────────────────────────────────────╮
│ 🎯  Flask REST API with SQLite users endpoint                │
│ # │ Tool         │ File / Command      │ Expected Outcome    │
│ 1 │ run_command  │ mkdir flask-api     │ Directory created   │
│ 2 │ run_command  │ cd flask-api        │ Working dir set     │
│ 3 │ write_file   │ app.py              │ Flask app written   │
│ 4 │ run_command  │ pip install flask   │ Flask installed     │
│ 5 │ run_command  │ python app.py       │ Server starts       │
╰──────────────────────────────────────────────────────────────╯

──────────────────────── Writer  Step 3 ──────────────────────
✍️  Writing → flask-api/app.py
   1 │ from flask import Flask, jsonify, request
   2 │ import sqlite3
   ...
✅  Written flask-api/app.py

──────────────────────── Executor  Step 4 ────────────────────
⚡  Running: pip install flask
╭─ stdout ───────────────────────────────────────────────────╮
│ Successfully installed flask-3.0.3 ...                     │
╰────────────────────────────────────────────────────────────╯
✅  Exit 0

──────────────────────── Execution Summary ───────────────────
 Event          │ Step │ Detail
 plan_created   │      │ Flask REST API with SQLite users endpoint
 exec_ok        │ 1    │ mkdir flask-api
 exec_cd        │ 2    │ flask-api
 write          │ 3    │ flask-api/app.py
 exec_ok        │ 4    │ pip install flask
 exec_ok        │ 5    │ python app.py

Files written: flask-api/app.py

🎉  Agent completed successfully!
```

### HITL in action

```
──────────────────────── HITL ────────────────────────────────
⚠️  Sensitive Command – Approval Required
╭─────────────────────────────────────────────────────────────╮
│ Command:   curl https://api.example.com/token               │
│ Reason:    Fetch auth token for deployment                  │
│ Expected:  Token written to .env                            │
╰─────────────────────────────────────────────────────────────╯
Approve this command? [y/N]: N
Decision: ❌ REJECTED
⏭  Skipping rejected command: curl https://api.example.com/token
```

---

## Why This Project Matters

Agentic AI systems that can reliably plan, write, and execute code are a core primitive for the next generation of developer tooling — from autonomous coding assistants to self-healing CI pipelines. This project demonstrates a working implementation of the key engineering challenges in that space:

- **Stateful multi-agent coordination** using LangGraph's directed graph model, where each agent reads from and writes to a shared typed state
- **Graceful failure handling** that doesn't require human intervention on every error — the system self-repairs and only escalates when it genuinely cannot recover
- **Safe autonomy** through systematic command classification, hard-blocking truly destructive operations, and giving humans meaningful control over sensitive actions rather than no control at all
- **Structured LLM outputs** via Pydantic models and `.with_structured_output()`, making the agents' decisions machine-readable and auditable rather than free-form strings
- **Incremental editing** that treats code as a living artifact — patching rather than regenerating, preserving context across the session

---

## Key Skills Demonstrated

| Skill | Where |
|---|---|
| **Multi-agent orchestration** | LangGraph `StateGraph` with 5 nodes and conditional routing |
| **Stateful agentic workflows** | `AgentState` TypedDict with LangGraph reducer annotations (`Annotated[List, add]`) |
| **Human-in-the-loop design** | `hitl_gate` node with regex-based command risk classification |
| **LLM-powered error recovery** | `repair` agent: receives broken code + stderr, returns corrected file |
| **Structured output parsing** | Pydantic v2 models (`ExecutionPlan`, `PlanStep`, `IncrementalPatch`) via `.with_structured_output()` |
| **Real-time streaming** | `GRAPH.stream()` in `updates` mode + Rich `Live` rendering |
| **Incremental code patching** | Search-and-replace diff with `difflib.SequenceMatcher` fuzzy fallback |
| **Audit trail / observability** | Append-only `execution_history` logged across every node |
| **CLI tooling** | Click `@cli.command` with both single-run and interactive REPL modes |
| **Production defensive patterns** | Hard-blocked commands, permission-error fallback on writes, 120s subprocess timeout |

---

## Developer Note

> **Note:** Developer can try with Closed source model for better performance like Claude's Opus 4.7.
>
> To switch models, update your `.env` file and use a compatible LangChain provider (e.g. `langchain-anthropic`):
>
> ```env
> ANTHROPIC_API_KEY=your_anthropic_api_key_here
> ```
>
> Then swap the LLM initialization in `final_v2.py`:
>
> ```python
> from langchain_anthropic import ChatAnthropic
>
> _MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")
>
> plan_llm   = ChatAnthropic(model=_MODEL, temperature=0.2).with_structured_output(ExecutionPlan)
> write_llm  = ChatAnthropic(model=_MODEL, temperature=0).with_structured_output(SingleFile)
> patch_llm  = ChatAnthropic(model=_MODEL, temperature=0).with_structured_output(IncrementalPatch)
> repair_llm = ChatAnthropic(model=_MODEL, temperature=0.3).with_structured_output(SingleFile)
> ```

---
