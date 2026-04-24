"""
Production-Grade Multi-Agent Coding System
==========================================
Features:
  - Planner → Writer → Executor pipeline (LangGraph)
  - Feedback loop with auto-retry on failures
  - Human-in-the-loop (HITL) approval for sensitive commands
  - Real-time streaming of agent reasoning & execution logs
  - Incremental / diff-based code patching
  - Rich terminal UX (colors, panels, spinners)
  - Error recovery with human escalation
"""

from __future__ import annotations

# ── stdlib ──────────────────────────────────────────────────────────────────
import difflib
import logging
import os
import re
import subprocess
import sys
import tempfile
from operator import add
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

# ── third-party ──────────────────────────────────────────────────────────────
import click
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── env & logging ─────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("coder_agent")

# ── Rich console with custom theme ────────────────────────────────────────────
THEME = Theme(
    {
        "agent.planner": "bold cyan",
        "agent.writer": "bold green",
        "agent.executor": "bold yellow",
        "agent.error": "bold red",
        "agent.success": "bold green",
        "agent.warning": "bold yellow",
        "agent.info": "dim white",
        "agent.hitl": "bold magenta",
        "agent.stream": "white",
        "agent.diff.add": "green",
        "agent.diff.remove": "red",
        "agent.diff.meta": "cyan",
    }
)
console = Console(theme=THEME)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PYDANTIC MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ToolArg(BaseModel):
    file_path: Optional[str] = Field(None, description="File path for the tool argument")
    command: Optional[str] = Field(None, description="Command to be executed")


class PlanStep(BaseModel):
    step_id: int = Field(..., description="Step id starting from 1")
    file_name: str = Field(..., description="File name of step")
    description: str = Field(..., description="Step description")
    tool_name: Literal["write_file", "run_command"] = Field(..., description="Tool to use")
    tool_args: ToolArg
    expected_outcome: str = Field(..., description="Expected outcome after step")
    is_incremental: bool = Field(
        False,
        description="True when this step modifies an existing file rather than creating a new one",
    )


class ExecutionPlan(BaseModel):
    goal: str = Field(..., description="Goal of the execution plan")
    steps: List[PlanStep] = Field(..., description="Ordered list of execution steps")


class SingleFile(BaseModel):
    file_name: str = Field(..., description="File name of generated code/content")
    content: str = Field(..., description="Full content of the file")


class IncrementalPatch(BaseModel):
    file_name: str = Field(..., description="File to patch")
    search_block: str = Field(..., description="Exact block to find and replace")
    replace_block: str = Field(..., description="Replacement block")
    reason: str = Field(..., description="Why this change is needed")


class ExecutionResult(BaseModel):
    status: Literal["success", "failure"] = Field(..., description="Execution status")
    stdout: Optional[str] = Field(None)
    stderr: Optional[str] = Field(None)
    retry_count: int = Field(0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentState(TypedDict):
    idea: str
    plan: ExecutionPlan
    generated_code: Optional[Annotated[List[Dict[str, str]], add]]
    execution_result: Optional[ExecutionResult]
    current_step_idx: int
    current_step_error: Optional[str]
    retry_count: int
    hitl_approved: Optional[bool]          # result of last HITL check
    execution_history: Annotated[List[Dict], add]  # full audit trail
    is_update_request: bool                # True for incremental edits


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
print("LLM MODEL: "+_MODEL)

plan_llm    = ChatGroq(model=_MODEL, temperature=0.2).with_structured_output(ExecutionPlan)
write_llm   = ChatGroq(model=_MODEL, temperature=0).with_structured_output(SingleFile)
patch_llm   = ChatGroq(model=_MODEL, temperature=0).with_structured_output(IncrementalPatch)
repair_llm  = ChatGroq(model=_MODEL, temperature=0.3).with_structured_output(SingleFile)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SENSITIVE COMMAND DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_HARD_BLOCKED = ["rm -rf /", "shutdown", "reboot", "format", "mkfs", "dd if="]

_SENSITIVE_PATTERNS = [
    r"\brm\b",            # file deletions
    r"\bchmod\b",         # permission changes
    r"\bchown\b",         # ownership changes
    r"\bsudo\b",          # privilege escalation
    r"\bcurl\b",          # network requests
    r"\bwget\b",          # network downloads
    r"\bnpm publish\b",   # package publishing
    r"\bgit push\b",      # remote pushes
    r"\beval\b",          # dynamic code eval
    r"\bexec\b",          # process replacement
    r">\s*/etc/",         # writing to system config
    r"\bkill\b",          # process termination
    r"\bpkill\b",
    r"\bsystemctl\b",
    r"\bservice\b",
    r"\biptables\b",
    r"\benv\b.*secret",   # env leaks
    r"subprocess\.call",  # nested subprocess
]
_SENSITIVE_RE = re.compile("|".join(_SENSITIVE_PATTERNS), re.IGNORECASE)


def is_hard_blocked(cmd: str) -> bool:
    return any(b in cmd.lower() for b in _HARD_BLOCKED)


def is_sensitive(cmd: str) -> bool:
    return bool(_SENSITIVE_RE.search(cmd))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INCREMENTAL PATCHING UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_patch(file_path: Path, patch: IncrementalPatch) -> tuple[bool, str]:
    """Apply a search-and-replace patch to an existing file.

    Returns (success, diff_text).
    """
    if not file_path.exists():
        return False, f"File {file_path} does not exist."

    original = file_path.read_text(encoding="utf-8")

    if patch.search_block not in original:
        # Fuzzy fallback: find best matching block via SequenceMatcher
        lines_orig = original.splitlines()
        lines_search = patch.search_block.splitlines()
        matcher = difflib.SequenceMatcher(None, lines_orig, lines_search)
        match = matcher.find_longest_match(0, len(lines_orig), 0, len(lines_search))
        if match.size < max(1, len(lines_search) // 2):
            return False, "search_block not found in file (fuzzy match too weak)."
        # reconstruct from matched lines
        matched_text = "\n".join(lines_orig[match.a : match.a + match.size])
        updated = original.replace(matched_text, patch.replace_block, 1)
    else:
        updated = original.replace(patch.search_block, patch.replace_block, 1)

    diff = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{file_path.name}",
            tofile=f"b/{file_path.name}",
        )
    )
    file_path.write_text(updated, encoding="utf-8")
    return True, "".join(diff)


def render_diff(diff_text: str) -> None:
    """Pretty-print a unified diff with Rich colours."""
    if not diff_text.strip():
        console.print("[agent.info]No changes in diff.[/]")
        return
    lines: list[Text] = []
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            lines.append(Text(line, style="agent.diff.meta"))
        elif line.startswith("+"):
            lines.append(Text(line, style="agent.diff.add"))
        elif line.startswith("-"):
            lines.append(Text(line, style="agent.diff.remove"))
        elif line.startswith("@@"):
            lines.append(Text(line, style="agent.diff.meta"))
        else:
            lines.append(Text(line))
    panel_content = Text("\n").join(lines)
    console.print(Panel(panel_content, title="📝 Diff", border_style="cyan"))


def detect_update_request(idea: str) -> bool:
    """Heuristic: detect if the user wants to modify existing code vs create new."""
    update_keywords = [
        "change", "update", "modify", "edit", "fix", "rename", "replace",
        "refactor", "adjust", "alter", "switch", "move", "add to", "remove from",
        "delete from", "set color", "set the", "make the",
    ]
    idea_lower = idea.lower()
    return any(kw in idea_lower for kw in update_keywords)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STREAMING HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def stream_text(label: str, text: str, style: str = "agent.stream") -> None:
    """Simulate token-level streaming output with Rich."""
    console.print(f"\n[{style}]{label}[/]")
    # In production with streaming-capable LLM, replace with actual stream events.
    # Here we chunk the text to give a live feel during LLM calls.
    words = text.split()
    buf = ""
    with Live(console=console, refresh_per_second=20) as live:
        for word in words:
            buf += word + " "
            live.update(Text(buf, style=style))
    console.print()  # newline after stream


def print_agent_header(agent: str, step: int | None = None) -> None:
    step_txt = f"  Step {step}" if step is not None else ""
    style_map = {
        "Planner": "agent.planner",
        "Writer": "agent.writer",
        "Executor": "agent.executor",
        "HITL": "agent.hitl",
        "Repair": "agent.error",
    }
    style = style_map.get(agent, "white")
    console.print(Rule(f"[{style}] {agent}{step_txt} [/{style}]", style=style))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT: PLANNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_PLANNER_SYSTEM = """You are a senior software engineer creating executable plans.

## STEP 1: Analyse requirements
## STEP 2: Identify directory structure — always create a project directory
## STEP 3: Plan directory navigation
CRITICAL: After creating any directory, ALWAYS have a step to CD into it.
## STEP 4: List all required files with dependencies
## STEP 5: Determine command sequence (install, init, build, run)

INCREMENTAL REQUESTS: If the user says "change", "update", "fix", etc., set
is_incremental=true on the relevant write_file step. Do NOT rewrite unrelated files.

QUALITY CHECKLIST:
✓ No file creation before CD into project directory
✓ No imports before package installation
✓ Step IDs sequential from 1
✓ Expected outcomes are measurable
✓ is_incremental correctly set

CRITICAL RULES:
- NO 'json' tool exists.
- For file writing, ALWAYS use 'write_file'.
- DO NOT create a virtual environment unless asked.
- DO NOT initialise git unless asked.
"""


def planner(state: AgentState) -> dict:
    idea = state["idea"]
    is_update = detect_update_request(idea)
    print_agent_header("Planner")

    with Progress(
        SpinnerColumn(),
        TextColumn("[agent.planner]Planning…"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("plan")
        plan = plan_llm.invoke(
            [
                SystemMessage(content=_PLANNER_SYSTEM),
                HumanMessage(content=f"Create a plan for: {idea}"),
            ]
        )

    # ── Pretty-print the plan ──────────────────────────────────────────────
    table = Table(title=f"🎯  {plan.goal}", box=box.ROUNDED, show_lines=True)
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Tool", style="yellow", width=12)
    table.add_column("File / Command", style="white")
    table.add_column("Expected Outcome", style="dim white")
    table.add_column("Incr?", style="magenta", width=6)

    for s in plan.steps:
        target = s.file_name if s.tool_name == "write_file" else (s.tool_args.command or "")
        table.add_row(
            str(s.step_id),
            s.tool_name,
            target[:60],
            s.expected_outcome[:60],
            "✓" if s.is_incremental else "",
        )

    console.print(table)
    console.print()

    return {
        "plan": plan,
        "is_update_request": is_update,
        "execution_history": [{"event": "plan_created", "goal": plan.goal}],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT: WRITER  (full write + incremental patch)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_WRITER_SYSTEM = """You are a senior software engineer.
Generate complete, working, production-quality code.
Rules:
- Full file content, no placeholders.
- All imports included.
- Proper error handling.
- No hardcoded secrets.
"""

_PATCH_SYSTEM = """You are a senior software engineer performing incremental code edits.
Given the current file content, the change instruction, and the expected outcome,
produce a minimal search-and-replace patch.
- search_block: exact verbatim substring currently in the file.
- replace_block: replacement (can be empty to delete).
- reason: one-sentence explanation.
Do NOT rewrite the whole file.
"""


def writer(state: AgentState) -> dict:
    plan = state["plan"]
    idx = state["current_step_idx"]
    step = plan.steps[idx]

    print_agent_header("Writer", idx + 1)

    file_path = Path(step.file_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    history_entry: Dict = {
        "event": "write",
        "step": idx + 1,
        "file": step.file_name,
        "incremental": step.is_incremental,
    }

    # ── Incremental patch ─────────────────────────────────────────────────
    if step.is_incremental and file_path.exists():
        console.print(f"[agent.writer]✂️  Incremental edit → {file_path}[/]")
        existing_content = file_path.read_text(encoding="utf-8")

        with Progress(SpinnerColumn(), TextColumn("[agent.writer]Generating patch…"), console=console, transient=True) as p:
            p.add_task("patch")
            patch = patch_llm.invoke(
                [
                    SystemMessage(content=_PATCH_SYSTEM),
                    HumanMessage(
                        content=(
                            f"File: {step.file_name}\n"
                            f"Current content:\n```\n{existing_content}\n```\n\n"
                            f"Instruction: {step.description}\n"
                            f"Expected outcome: {step.expected_outcome}"
                        )
                    ),
                ]
            )

        success, diff_text = apply_patch(file_path, patch)
        if success:
            render_diff(diff_text)
            stream_text("  [agent.writer]Patch reasoning:[/]", patch.reason, style="agent.info")
            console.print(f"[agent.success]✅  Patched {file_path}[/]")
            history_entry["status"] = "patch_ok"
        else:
            console.print(
                Panel(
                    f"[agent.error]Patch failed: {diff_text}\nFalling back to full rewrite.[/]",
                    border_style="red",
                )
            )
            # Fall through to full write below
            step.is_incremental = False
            history_entry["status"] = "patch_fallback"

    # ── Full file write ───────────────────────────────────────────────────
    if not step.is_incremental or not file_path.exists():
        console.print(f"[agent.writer]✍️  Writing → {file_path}[/]")

        with Progress(SpinnerColumn(), TextColumn("[agent.writer]Generating code…"), console=console, transient=True) as p:
            p.add_task("write")
            generated = write_llm.invoke(
                [
                    SystemMessage(content=_WRITER_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Description: {step.description}\n"
                            f"file_name: {step.file_name}\n"
                            f"expected_output: {step.expected_outcome}"
                        )
                    ),
                ]
            )

        content = generated.content
        resolved_path = _safe_write(file_path, content)
        # Stream a preview of the generated code
        lang = resolved_path.suffix.lstrip(".") or "text"
        preview = content if len(content) < 1200 else content[:1200] + "\n# … (truncated)"
        console.print(Syntax(preview, lang, theme="monokai", line_numbers=True))
        console.print(f"[agent.success]✅  Written {resolved_path}[/]")

        history_entry["status"] = "full_write_ok"
        return {
            "generated_code": [{str(resolved_path): content}],
            "current_step_idx": idx + 1,
            "execution_history": [history_entry],
        }

    return {
        "generated_code": [{str(file_path): file_path.read_text(encoding="utf-8")}],
        "current_step_idx": idx + 1,
        "execution_history": [history_entry],
    }


def _safe_write(file_path: Path, content: str) -> Path:
    """Write content, falling back to cwd or tempdir on permission errors."""
    for candidate in [file_path, Path.cwd() / file_path.name, Path(tempfile.gettempdir()) / file_path.name]:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(content, encoding="utf-8")
            return candidate
        except PermissionError:
            continue
    raise PermissionError(f"Cannot write {file_path} – no writable location found.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT: HITL GATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def hitl_gate(state: AgentState) -> dict:
    """Human-in-the-loop approval for sensitive or destructive commands."""
    plan = state["plan"]
    idx = state["current_step_idx"]
    step = plan.steps[idx]
    cmd = (step.tool_args.command or "").strip()

    print_agent_header("HITL")

    console.print(
        Panel(
            f"[bold]Command:[/bold] [yellow]{cmd}[/yellow]\n"
            f"[bold]Reason:[/bold]  {step.description}\n"
            f"[bold]Expected:[/bold] {step.expected_outcome}",
            title="⚠️  Sensitive Command – Approval Required",
            border_style="magenta",
        )
    )

    approved = Confirm.ask("[agent.hitl]Approve this command?[/]", default=False)

    console.print(
        f"[agent.hitl]Decision: {'✅ APPROVED' if approved else '❌ REJECTED'}[/]"
    )

    return {
        "hitl_approved": approved,
        "execution_history": [
            {"event": "hitl", "step": idx + 1, "command": cmd, "approved": approved}
        ],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT: EXECUTOR  (with feedback loop)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAX_RETRIES = 3


def executor(state: AgentState) -> dict:
    plan = state["plan"]
    idx = state["current_step_idx"]
    step = plan.steps[idx]
    cmd = (step.tool_args.command or "").strip()
    retry = state.get("retry_count", 0)

    # HITL: rejected command → skip with failure recorded
    if state.get("hitl_approved") is False:
        console.print(f"[agent.warning]⏭  Skipping rejected command: {cmd}[/]")
        return {
            "execution_result": ExecutionResult(status="failure", stderr="Rejected by user (HITL)", retry_count=retry),
            "current_step_idx": idx + 1,
            "retry_count": 0,
            "hitl_approved": None,
            "current_step_error": "User rejected command via HITL.",
            "execution_history": [{"event": "exec_skipped", "step": idx + 1, "command": cmd}],
        }

    print_agent_header("Executor", idx + 1)

    if not cmd:
        raise ValueError(f"Step {idx + 1}: no command provided for executor.")

    # cd handling
    if cmd.startswith("cd "):
        dir_name = cmd.split(" ", 1)[1].strip()
        try:
            os.chdir(dir_name)
            console.print(f"[agent.executor]📂  cd → {Path.cwd()}[/]")
            return {
                "execution_result": ExecutionResult(status="success", stdout=f"cd {dir_name}"),
                "current_step_idx": idx + 1,
                "retry_count": 0,
                "hitl_approved": None,
                "execution_history": [{"event": "exec_cd", "step": idx + 1, "dir": dir_name}],
            }
        except FileNotFoundError as exc:
            console.print(f"[agent.error]Directory not found: {exc}[/]")
            return {
                "execution_result": ExecutionResult(status="failure", stderr=str(exc)),
                "current_step_idx": idx + 1,
                "retry_count": 0,
                "current_step_error": str(exc),
                "execution_history": [{"event": "exec_cd_fail", "step": idx + 1, "error": str(exc)}],
            }

    console.print(f"[agent.executor]⚡  Running:[/] [bold]{cmd}[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[agent.executor]Executing…"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("exec")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            console.print("[agent.error]⏰  Command timed out.[/]")
            return {
                "execution_result": ExecutionResult(status="failure", stderr="Timeout after 120s", retry_count=retry),
                "current_step_idx": idx + 1,
                "retry_count": retry + 1,
                "current_step_error": "Timeout",
                "execution_history": [{"event": "exec_timeout", "step": idx + 1}],
            }

    # Stream stdout
    if result.stdout.strip():
        console.print(Panel(result.stdout.strip(), title="stdout", border_style="green"))
    if result.stderr.strip():
        console.print(Panel(result.stderr.strip(), title="stderr", border_style="red"))

    if result.returncode == 0:
        console.print("[agent.success]✅  Exit 0[/]")
        return {
            "execution_result": ExecutionResult(status="success", stdout=result.stdout, retry_count=retry),
            "current_step_idx": idx + 1,
            "retry_count": 0,
            "current_step_error": None,
            "hitl_approved": None,
            "execution_history": [{"event": "exec_ok", "step": idx + 1, "command": cmd}],
        }
    else:
        console.print(f"[agent.error]❌  Exit {result.returncode}[/]")
        return {
            "execution_result": ExecutionResult(
                status="failure",
                stdout=result.stdout,
                stderr=result.stderr,
                retry_count=retry,
            ),
            "current_step_idx": idx,          # keep on same step for retry
            "retry_count": retry + 1,
            "current_step_error": result.stderr,
            "execution_history": [
                {"event": "exec_fail", "step": idx + 1, "command": cmd, "stderr": result.stderr[:300]}
            ],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT: REPAIR  (LLM-assisted error recovery)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def repair(state: AgentState) -> dict:
    """Attempt LLM-assisted repair of the last failing file."""
    plan = state["plan"]
    idx = state["current_step_idx"]
    step = plan.steps[idx]
    error = state.get("current_step_error", "Unknown error")
    retry = state.get("retry_count", 0)

    print_agent_header("Repair")
    console.print(
        Panel(
            f"[bold red]Error:[/bold red] {error}\n"
            f"[dim]Retry attempt {retry}/{MAX_RETRIES}[/dim]",
            border_style="red",
        )
    )

    # Find last generated content for this file
    last_content = ""
    for code_dict in reversed(state.get("generated_code") or []):
        if step.file_name in code_dict:
            last_content = code_dict[step.file_name]
            break

    if not last_content:
        console.print("[agent.error]No prior content to repair – skipping step.[/]")
        return {
            "current_step_idx": idx + 1,
            "retry_count": 0,
            "current_step_error": None,
            "execution_history": [{"event": "repair_skip", "step": idx + 1}],
        }

    with Progress(SpinnerColumn(), TextColumn("[agent.error]Repairing…"), console=console, transient=True) as p:
        p.add_task("repair")
        repaired = repair_llm.invoke(
            [
                SystemMessage(content="You are a debugging expert. Fix the provided code based on the error."),
                HumanMessage(
                    content=(
                        f"File: {step.file_name}\n"
                        f"Error: {error}\n"
                        f"Broken code:\n```\n{last_content}\n```\n"
                        f"Return the fully corrected file."
                    )
                ),
            ]
        )

    resolved = _safe_write(Path(step.file_name), repaired.content)
    console.print(f"[agent.success]🔧  Repaired and wrote {resolved}[/]")

    return {
        "generated_code": [{str(resolved): repaired.content}],
        "current_step_error": None,
        "execution_history": [{"event": "repaired", "step": idx + 1}],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROUTING LOGIC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def router(state: AgentState) -> str:
    """Central router: determines next node after planner/writer/executor."""
    plan = state["plan"]
    idx = state["current_step_idx"]

    if idx >= len(plan.steps):
        return "__end__"

    step = plan.steps[idx]

    if step.tool_name == "write_file":
        return "writer"
    elif step.tool_name == "run_command":
        cmd = (step.tool_args.command or "").strip()
        if is_sensitive(cmd):
            return "hitl_gate"
        return "executor"
    return "__end__"


def post_executor_router(state: AgentState) -> str:
    """After executor: handle failure → repair → retry or escalate."""
    result = state.get("execution_result")
    retry = state.get("retry_count", 0)
    idx = state["current_step_idx"]
    plan = state["plan"]

    if result and result.status == "failure":
        if retry < MAX_RETRIES:
            # Check if there's a file that could be repaired
            step = plan.steps[idx]
            if step.tool_name == "write_file":
                return "repair"
            # For commands, retry directly
            console.print(
                f"[agent.warning]↩  Retrying step {idx + 1} (attempt {retry + 1}/{MAX_RETRIES})…[/]"
            )
            return "executor"
        else:
            # Human escalation
            console.print(
                Panel(
                    f"[bold red]Step {idx + 1} failed {MAX_RETRIES} times.[/bold red]\n"
                    f"Error: {state.get('current_step_error', 'unknown')}\n\n"
                    "Please review and press Enter to skip or Ctrl+C to abort.",
                    title="🆘  Human Escalation Required",
                    border_style="red",
                )
            )
            try:
                Prompt.ask("[agent.hitl]Skip this step and continue?[/]", default="yes")
            except (EOFError, KeyboardInterrupt):
                sys.exit(1)
            # Skip the failing step
            state["current_step_idx"] += 1
            return router(state)

    return router(state)


def post_hitl_router(state: AgentState) -> str:
    """After HITL gate: approved → executor, rejected → skip via router."""
    if state.get("hitl_approved"):
        return "executor"
    # Rejected: advance index and route normally
    state["current_step_idx"] += 1
    return router(state)


def post_repair_router(state: AgentState) -> str:
    """After repair: re-run the executor for the same step."""
    return "executor"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LANGGRAPH GRAPH CONSTRUCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("planner",   planner)
    builder.add_node("writer",    writer)
    builder.add_node("hitl_gate", hitl_gate)
    builder.add_node("executor",  executor)
    builder.add_node("repair",    repair)

    # Edges
    builder.add_edge(START, "planner")

    _choices = {"writer": "writer", "executor": "executor", "hitl_gate": "hitl_gate", "__end__": END}

    builder.add_conditional_edges("planner",   router,              _choices)
    builder.add_conditional_edges("writer",    router,              _choices)
    builder.add_conditional_edges("hitl_gate", post_hitl_router,    {**_choices, "executor": "executor"})
    builder.add_conditional_edges("executor",  post_executor_router, {**_choices, "repair": "repair", "executor": "executor"})
    builder.add_conditional_edges("repair",    post_repair_router,   {"executor": "executor"})

    return builder.compile()


GRAPH = build_graph()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TERMINAL UI  (Rich + Click)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_banner() -> None:
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          🤖  Autonomous Coder Agent  v2.0                    ║
║          Planner → Writer → Executor  (LangGraph)            ║
╚══════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold cyan")
    console.print(
        Columns(
            [
                "[cyan]✓[/] Feedback loop (auto-retry)",
                "[magenta]✓[/] Human-in-the-loop (HITL)",
                "[green]✓[/] Streaming output",
                "[yellow]✓[/] Incremental patching",
            ]
        )
    )
    console.print()


def print_summary(final_state: AgentState) -> None:
    console.print(Rule("[bold green]Execution Summary[/bold green]", style="green"))
    history = final_state.get("execution_history", [])

    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Event", style="cyan")
    table.add_column("Step", style="yellow")
    table.add_column("Detail", style="white")

    for entry in history:
        event = entry.get("event", "")
        step  = str(entry.get("step", ""))
        detail = (
            entry.get("command", "")
            or entry.get("file", "")
            or entry.get("error", "")
            or entry.get("dir", "")
            or entry.get("goal", "")
        )
        table.add_row(event, step, detail[:80])

    console.print(table)

    generated = final_state.get("generated_code") or []
    if generated:
        files = sorted({k for d in generated for k in d})
        console.print(f"\n[bold]Files written:[/bold] {', '.join(files)}")

    console.print(Rule(style="green"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@click.group()
def cli():
    """Autonomous Coder Agent – multi-agent LangGraph coding system."""


@cli.command("run")
@click.argument("idea", required=False)
@click.option("--stream/--no-stream", default=True, help="Stream agent output (default: on)")
@click.option("--max-retries", default=MAX_RETRIES, show_default=True, help="Max executor retries per step")
def run(idea: str | None, stream: bool, max_retries: int) -> None:
    """Run the coding agent with IDEA (or prompt interactively)."""
    print_banner()

    if not idea:
        idea = Prompt.ask("[bold cyan]💡 What would you like to build or change?[/]")

    if not idea.strip():
        console.print("[red]No idea provided. Exiting.[/]")
        sys.exit(1)

    console.print(f"\n[bold]Task:[/bold] {idea}\n")

    initial_state: AgentState = {
        "idea": idea,
        "plan": None,           # type: ignore[assignment]
        "generated_code": [],
        "execution_result": None,
        "current_step_idx": 0,
        "current_step_error": None,
        "retry_count": 0,
        "hitl_approved": None,
        "execution_history": [],
        "is_update_request": False,
    }

    try:
        if stream:
            # Stream node-by-node events
            final_state = None
            for event in GRAPH.stream(initial_state, stream_mode="updates"):
                # Each event is {node_name: state_patch}; already printed inside agents
                for node, patch in event.items():
                    if node not in ("__start__",):
                        pass  # agents print their own output
                # Keep last state for summary
                final_state = event
            # Re-invoke to get the final merged state
            final_state = GRAPH.invoke(initial_state)
        else:
            final_state = GRAPH.invoke(initial_state)

        print_summary(final_state)
        console.print("\n[bold green]🎉  Agent completed successfully![/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]⛔  Interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print_exception()
        console.print(f"\n[bold red]Fatal error: {exc}[/bold red]")
        sys.exit(1)


@cli.command("interactive")
def interactive() -> None:
    """Interactive REPL – send multiple tasks in sequence."""
    print_banner()
    console.print("[dim]Type 'exit' or 'quit' to stop.[/dim]\n")

    while True:
        try:
            idea = Prompt.ask("[bold cyan]💡 Task[/]")
        except (EOFError, KeyboardInterrupt):
            break

        if idea.strip().lower() in ("exit", "quit", "q"):
            break

        initial_state: AgentState = {
            "idea": idea,
            "plan": None,           # type: ignore[assignment]
            "generated_code": [],
            "execution_result": None,
            "current_step_idx": 0,
            "current_step_error": None,
            "retry_count": 0,
            "hitl_approved": None,
            "execution_history": [],
            "is_update_request": False,
        }

        try:
            final_state = GRAPH.invoke(initial_state)
            print_summary(final_state)
        except Exception as exc:
            console.print_exception()
            console.print(f"[bold red]Error: {exc}[/bold red]")

    console.print("\n[cyan]Goodbye! 👋[/cyan]")


# ── Direct script entry ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Fallback: behave like the original script when run directly
    if len(sys.argv) == 1:
        print_banner()
        idea = Prompt.ask("[bold cyan]💡 What would you like to build?[/]")
        initial_state: AgentState = {
            "idea": idea,
            "plan": None,           # type: ignore[assignment]
            "generated_code": [],
            "execution_result": None,
            "current_step_idx": 0,
            "current_step_error": None,
            "retry_count": 0,
            "hitl_approved": None,
            "execution_history": [],
            "is_update_request": False,
        }
        final_state = GRAPH.invoke(initial_state)
        print_summary(final_state)
    else:
        cli()
