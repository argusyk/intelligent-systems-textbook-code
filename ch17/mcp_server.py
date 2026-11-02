# MCP-сервер «ДСУ-Експерта»: надає інструменти діагностики й пошуку в керівництві
# будь-якому MCP-сумісному клієнту (Claude Desktop, Claude Code, IDE тощо).
# Встановлення: pip install mcp ; запуск: python mcp_server.py (транспорт stdio).
# Без пакета mcp програма лише перевіряє функції-інструменти локально.
from agent import get_parameters, run_diagnosis
from rag import Index, build

INDEX = Index(build(2, 1))

def diagnose(unit: str) -> dict:
    """Повертає параметри ДСУ за бортовим номером і гіпотези експертної системи з CF."""
    p = get_parameters(unit)
    return {"params": p, **run_diagnosis(p)}

def search_manual(query: str, k: int = 3) -> list[dict]:
    """Шукає фрагменти керівництва з ТО ДСУ; повертає розділ і текст."""
    return [{"section": s, "text": t} for s, t, _ in INDEX.search(query, k=k)]

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None

if __name__ == "__main__":
    if FastMCP is None:
        print("пакет mcp не встановлено; локальна перевірка інструментів:")
        print(" diagnose:", diagnose("UR-PSA"))
        print(" search_manual:", search_manual("тиск масла норма", k=1))
    else:
        mcp = FastMCP("dsu-expert")
        mcp.tool()(diagnose)                     # опис і схема беруться з підпису та docstring
        mcp.tool()(search_manual)
        mcp.run()                                # stdio: клієнт запускає сервер як підпроцес
