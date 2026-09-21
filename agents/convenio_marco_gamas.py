"""
convenio_marco_gamas.py

Carga la tabla de especificaciones mínimas por Gama del Convenio Marco
vigente desde diccionarios/convenio_marco_gamas.json (config estática, sin
DB ni red -- se carga una vez en memoria, mismo patrón que gama_service.py
con diccionarios/gama_procesador.json). El dato real vive en el JSON, no
acá -- este módulo es solo el loader + las funciones de consulta, para que
alguien pueda actualizar la tabla (ej. si cambia el convenio marco) editando
un archivo de datos, sin tocar código Python.

NO confundir con gama_service.py: ese clasifica LÍNEAS DE PROCESADOR en 4
niveles de rendimiento (Básica/Media/Alta/Premium) para agrupar equivalentes
cross-marca; este archivo consulta la tabla OFICIAL y CONTRACTUAL de
"Gama 2/3/4" del convenio marco, con pisos mínimos de RAM/procesador/disco/
GPU/pantalla y un techo de precio en USD neto por tipo de equipo.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

_DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diccionarios", "convenio_marco_gamas.json")

_data: Optional[Dict] = None
_gamas_by_key: Optional[Dict[tuple, Dict[str, Any]]] = None
_trigger_re: Optional[re.Pattern] = None


def _load() -> Dict:
    global _data, _gamas_by_key
    if _data is None:
        try:
            with open(_DICT_PATH, "r", encoding="utf-8") as f:
                _data = json.load(f)
        except Exception as e:
            logging.warning(f"[convenio_marco_gamas] No se pudo cargar {_DICT_PATH}: {e}")
            _data = {"gamas": [], "sinonimos_gama": {}}
        _gamas_by_key = {
            (row["tipo_equipo"], row["gama"]): row
            for row in _data.get("gamas", [])
        }
    return _data


def get_specs(tipo_equipo: Optional[str], gama: int) -> Optional[Dict[str, Any]]:
    """Specs mínimas oficiales para (tipo_equipo, gama), o None si esa
    combinación no está definida en el Anexo A (ej. gama 1, o AIO gama 4)."""
    _load()
    if not tipo_equipo:
        return None
    return _gamas_by_key.get((tipo_equipo.strip().title(), gama))


def resolve_gama(texto: str) -> Optional[int]:
    """Traduce una mención en lenguaje natural ("gama alta", "gama 2") al
    número de gama oficial. None si no matchea ningún sinónimo conocido."""
    data = _load()
    return data.get("sinonimos_gama", {}).get(texto.strip().lower())


def mentions_gama(texto: str) -> bool:
    """Detección barata (una sola regex, sin LLM ni embeddings) de si un
    mensaje menciona una gama -- para decidir si vale la pena inyectar la
    tabla completa en el prompt de ese turno o no (ver format_for_prompt).
    Solo matchea "gama" + un sinónimo textual (alta/media/básica/premium/
    workstation/entrada/baja/avanzada/intermedia) -- los sinónimos puramente
    numéricos ("1","2","3","4") se excluyen a propósito: harían falso
    positivo con cualquier mención de cantidad ("necesito 2 laptops")."""
    global _trigger_re
    if _trigger_re is None:
        data = _load()
        palabras = sorted({
            k for k in data.get("sinonimos_gama", {})
            if not k.isdigit()
        }, key=len, reverse=True)
        pattern = r"\b(" + "|".join(re.escape(p) for p in palabras) + r")\b"
        _trigger_re = re.compile(pattern, re.IGNORECASE)
    return bool(_trigger_re.search(texto or ""))


def format_gama_summary(gama: int) -> Optional[str]:
    """Resumen corto (procesador + RAM + disco [+ GPU]) de una gama, sin
    pantalla ni tipo_equipo -- para citar el nivel de rendimiento en
    secciones del prompt que no necesitan la tabla completa por tipo de
    equipo (ver "ESPECIFICACIONES MÍNIMAS SEGÚN USO" en ficha_agent.py,
    petición de que esa sección use la misma Gama oficial como base en vez
    de números inventados aparte). Usa la primera fila que matchee esa
    gama -- procesador/núcleos/RAM/disco son iguales entre Laptop/Desktop/
    AIO para una misma gama, solo cambian pantalla o si aplica GPU."""
    data = _load()
    row = next((r for r in data.get("gamas", []) if r["gama"] == gama), None)
    if not row:
        return None
    parts = [
        f"procesador {'/'.join(row['linea_procesador'])} o superior ({row['nucleos_min']}+ núcleos)",
        f"{row['total_ram_gb']} GB RAM",
        f"{row['total_almacenamiento_gb']} GB SSD NVMe",
    ]
    if row.get("tiene_gpu_dedicada"):
        parts.append(f"GPU dedicada (mín. {row.get('gpu_vram_min_gb', '?')} GB VRAM)")
    return ", ".join(parts)


def format_for_prompt() -> str:
    """Tabla en texto plano para inyectar en el system prompt del agente --
    se genera desde el JSON (fuente única de verdad), no se escribe a mano
    aparte, así el prompt nunca queda desalineado con los datos reales."""
    data = _load()
    lines: List[str] = []
    for row in data.get("gamas", []):
        parts = [
            f"procesador {'/'.join(row['linea_procesador'])} o superior ({row['nucleos_min']}+ núcleos)",
            f"{row['total_ram_gb']} GB RAM",
            f"{row['total_almacenamiento_gb']} GB SSD NVMe",
        ]
        pantalla = row.get("pantalla_pulgadas")
        if pantalla:
            if "max" in pantalla:
                parts.append(f"pantalla {pantalla['min']}\"-{pantalla['max']}\"")
            else:
                parts.append(f"pantalla {pantalla['min']}\" o superior")
        if row.get("tiene_gpu_dedicada"):
            parts.append(f"GPU dedicada (mín. {row.get('gpu_vram_min_gb', '?')} GB VRAM)")
        parts.append(row["sistema_operativo"])
        lines.append(f"- {row['tipo_equipo']} Gama {row['gama']}: " + ", ".join(parts))
    return "\n".join(lines)
