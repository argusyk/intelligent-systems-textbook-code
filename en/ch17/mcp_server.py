# APU-Expert MCP server: provides diagnostic and manual search tools
# to any MCP-compatible client (Claude Desktop, Claude Code, IDEs, etc.).
# Installation: pip install mcp ; run: python mcp_server.py (stdio transport).
# Without the mcp package the program only checks the tool functions locally.
from agent import get_parameters, run_diagnosis
from rag import Index, build

INDEX = Index(build(2, 1))

def diagnose(unit: str) -> dict:
    """Returns the APU parameters for a tail number and the expert system's hypotheses with CF."""
    p = get_parameters(unit)
    return {"params": p, **run_diagnosis(p)}

def search_manual(query: str, k: int = 3) -> list[dict]:
    """Searches the APU maintenance manual; returns the section and the text."""
    return [{"section": s, "text": t} for s, t, _ in INDEX.search(query, k=k)]

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None

if __name__ == "__main__":
    if FastMCP is None:
        print("the mcp package is not installed; local check of the tools:")
        print(" diagnose:", diagnose("UR-PSA"))
        print(" search_manual:", search_manual("normal oil pressure", k=1))
    else:
        mcp = FastMCP("apu-expert")
        mcp.tool()(diagnose)             # the description and schema come from the signature and docstring
        mcp.tool()(search_manual)
        mcp.run()                        # stdio: the client starts the server as a subprocess
