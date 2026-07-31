"""
processor_family.py

Extrae la línea/familia canónica de un procesador a partir de texto libre
(ej. "Intel Core i5-1335U" → "Intel Core i5").

Usado por:
  - ficha_agent.py: para autocompletar linea_procesador cuando el usuario
    da un modelo exacto (vocabulario alineado al modelo LGBM).
  - services/cm_service.py: para comparar la ficha contra el texto crudo,
    heterogéneo, de procesador_principal en el catálogo de Convenio Marco.
"""

import re
from typing import Optional

# Orden importa: los patrones más específicos van primero.
LINEA_RULES = [
    (r"Core\s+Ultra\s+9",     "Intel Core Ultra 9"),
    (r"Core\s+Ultra\s+7",     "Intel Core Ultra 7"),
    (r"Core\s+Ultra\s+5",     "Intel Core Ultra 5"),
    (r"Core\s+i9",            "Intel Core i9"),
    (r"Core\s+i7",            "Intel Core i7"),
    (r"Core\s+i5",            "Intel Core i5"),
    (r"Core\s+i3",            "Intel Core i3"),
    # Rebrand 2023: Intel Core 3/5/7/9 sin "i" — familia distinta de Core i3/i5/i7/i9,
    # no mezclar (arquitectura y generación diferentes).
    (r"Core\s+9\b",           "Intel Core 9"),
    (r"Core\s+7\b",           "Intel Core 7"),
    (r"Core\s+5\b",           "Intel Core 5"),
    (r"Core\s+3\b",           "Intel Core 3"),
    (r"\bCeleron\b",          "Intel Celeron"),
    (r"\bPentium\b",          "Intel Pentium"),
    (r"\bXeon\b",             "Intel Xeon"),
    (r"\bN1\d{2}\b",          "Intel N-Series"),  # N100, N150, etc.
    (r"Ryzen\s+AI\s+9",       "AMD Ryzen AI 9"),
    (r"Ryzen\s+AI\s+7",       "AMD Ryzen AI 7"),
    (r"Ryzen\s+AI\s+5",       "AMD Ryzen AI 5"),
    (r"Ryzen\s+(?:R\s*)?9",   "AMD Ryzen 9"),
    (r"Ryzen\s+(?:R\s*)?7",   "AMD Ryzen 7"),
    (r"Ryzen\s+(?:R\s*)?5",   "AMD Ryzen 5"),
    (r"Ryzen\s+(?:R\s*)?3",   "AMD Ryzen 3"),
    (r"\bAthlon\b",           "AMD Athlon"),
    (r"Apple\s+M5\s+Max",     "Apple M5 Max"),
    (r"Apple\s+M5\s+Pro",     "Apple M5 Pro"),
    (r"Apple\s+M5\b",         "Apple M5"),
    (r"Apple\s+M4\s+Max",     "Apple M4 Max"),
    (r"Apple\s+M4\s+Pro",     "Apple M4 Pro"),
    (r"Apple\s+M4\b",         "Apple M4"),
    (r"Apple\s+M3\s+Pro",     "Apple M3 Pro"),
    (r"Apple\s+M3\b",         "Apple M3"),
    (r"Apple\s+M[12]\b",      "Apple M-Series"),
    (r"Apple\s+M\d",          "Apple M-Series"),
    # Catálogo Convenio Marco usa "CHIP M5" en vez de "Apple M5"
    (r"\bChip\s+M5\s+Max\b",  "Apple M5 Max"),
    (r"\bChip\s+M5\s+Pro\b",  "Apple M5 Pro"),
    (r"\bChip\s+M5\b",        "Apple M5"),
    (r"\bChip\s+M4\s+Max\b",  "Apple M4 Max"),
    (r"\bChip\s+M4\s+Pro\b",  "Apple M4 Pro"),
    (r"\bChip\s+M4\b",        "Apple M4"),
    (r"\bChip\s+A\d+\b",      "Apple A-Series"),
    (r"Snapdragon\s+X\s+Plus","Snapdragon X Plus"),
    (r"Snapdragon\s+X\s+Elite","Snapdragon X Elite"),
    (r"Snapdragon\s+X\b",     "Snapdragon X"),
    (r"Kompanio",             "MediaTek Kompanio"),
]

_COMPILED_RULES = [(re.compile(pattern, re.IGNORECASE), canonical) for pattern, canonical in LINEA_RULES]


def extract_linea_procesador(procesador: Optional[str]) -> Optional[str]:
    if not procesador:
        return None
    for pattern, canonical in _COMPILED_RULES:
        if pattern.search(procesador):
            return canonical
    return None
