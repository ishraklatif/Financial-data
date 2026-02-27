"""
canonical_map.py — Single source of truth for ticker → canonical name.
"""
import re
from typing import Dict, List

CANONICAL: Dict[str, str] = {
    "^AXJO":"AXJO", "^GSPC":"GSPC", "^FTSE":"FTSE", "^N225":"N225",
    "^HSI":"HSI", "^VIX":"VIX", "^VVIX":"VVIX", "^MOVE":"MOVE", "^BDI":"BDI",
    "000001.SS":"SSE", "000300.SS":"CSI300",
    "DX-Y.NYB":"DXY", "AUDUSD=X":"AUDUSD", "AUDJPY=X":"AUDJPY", "AUDCNY=X":"AUDCNY",
    "GC=F":"GOLD", "BZ=F":"OIL", "CL=F":"OIL",
    "HG=F":"COPPER", "SI=F":"SILVER", "TIO=F":"IRON",
}

def canonical(s: str) -> str:
    return CANONICAL.get(s, re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_"))

def safe_name(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_")

def group_by_canonical(symbols: List[str]) -> Dict[str, List[str]]:
    g: Dict[str, List[str]] = {}
    for s in symbols:
        g.setdefault(canonical(s), []).append(s)
    return g
