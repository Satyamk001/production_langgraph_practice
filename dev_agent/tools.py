import sys
import subprocess
from langchain_core.tools import tool
from dev_agent.config import WORKSPACE

def safe_path(name : str) -> str:
    """Sirf workspace ke andar ki files allow karo"""
    root = WORKSPACE.resolve()
    p = (root / name).resolve()
    
    if(root not in p.parents):
        raise ValueError(f"{name} is outside allowd workspace")
    return p

@tool
def list_files() -> str:
    """List all the files under workspace directory"""
    names = sorted(p.name for p in WORKSPACE.iterdir())
    return "\n".join(names) or "Workspace is empty"

@tool
def read_file(name: str) -> str:
    """Read a txt file from workspace"""
    try:
        return safe_path(name).read_text()
    except FileNotFoundError:
        return f"Error: '{name}' does not exist in workspace. check name using list_files"
    except ValueError as e:
        return f"Error: {e}"
    
@tool 
def write_file(name: str, content: str) -> str:
    """Write text content to a file in the workspace (overwrite if it exists)"""
    try:
        safe_path(name).write_text(content)
        return f"OK: '{name}' written ({len(content)} chars)"
    except ValueError as e:
        return f"Error: {e}"
    
@tool 
def run_pytest(name : str="") -> str:
    """Run pytest inside the workspace (optionally one test file). Returns the pytest output."""
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short"] + ([name] if name else [])
    try:
        r = subprocess.run(cmd, cwd=WORKSPACE, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr)[-3000:] or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: pytest did not end in 60 sec (timeout)"
