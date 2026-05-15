"""
lgbm_price_service.py

Estima precios del mercado externo (SoloTodo) usando modelos LightGBM agg_bal.

- Carga los modelos P25, P75 y Mean para Notebooks y All-in-One.
- P25/P75 predicen en log-space → se aplica exp() para convertir a CLP.
- Mean predice directamente en CLP.
- Tipo de equipo: Laptop → Notebooks, AIO → All-in-One. Desktop y otros → None.
"""

import os
import re
import pickle
import logging
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "lgbm",
)

TIPO_TO_CAT = {
    "Laptop": "notebooks",
    "AIO":    "all_in_one",
}

FEATURE_COLS = [
    "total_ram_gb", "total_almacenamiento_gb", "nucleos_procesador",
    "hilos_procesador", "frecuencia_turbo_procesador_mhz", "frecuencia_ram_mhz",
    "pantalla_pulgadas",
    "marca", "linea_producto", "linea_procesador", "generacion_procesador",
    "tecnologia_ram", "tecnologia_disco_principal", "tipo_configuracion_discos",
    "gpu_dedicada_nombre", "tecnologia_gpu_principal", "sistema_operativo",
    "wifi_generacion", "resolucion_pantalla_pixeles",
    "year", "month", "week_of_year", "days_since_start", "semester_idx",
]

CAT_COLS = [
    "marca", "linea_producto", "linea_procesador", "generacion_procesador",
    "tecnologia_ram", "tecnologia_disco_principal", "tipo_configuracion_discos",
    "gpu_dedicada_nombre", "tecnologia_gpu_principal", "sistema_operativo",
    "wifi_generacion", "resolucion_pantalla_pixeles",
]

_START_DATE = date(2022, 1, 1)

# Normalización de tecnologia_disco_principal al vocabulario del modelo
_DISCO_NORM = {
    "NVMe SSD":  "SSD",
    "SATA SSD":  "SSD",
    "SSD":       "SSD",
    "mSATA":     "SSD",
    "HDD":       "HDD",
    "eMMC":      "eMMC",
}


def _derive_gpu_tech(gpu_nombre: Optional[str]) -> str:
    if not gpu_nombre:
        return "Integrated"
    name = str(gpu_nombre).strip()
    if name.lower() in ("sin gpu dedicada", "none", "nan", ""):
        return "Integrated"
    up = name.upper()
    if any(k in up for k in ("NVIDIA", "RTX", "GTX", "QUADRO", "GEFORCE")):
        return "NVIDIA"
    if any(k in up for k in ("AMD", "RADEON", "RX ")):
        return "AMD"
    if any(k in up for k in ("INTEL", "ARC", "IRIS")):
        return "Intel"
    return "Integrated"


def _temporal(ref: date) -> dict:
    days = (ref - _START_DATE).days
    sem = (ref.year - 2022) * 2 + (0 if ref.month <= 6 else 1)
    return {
        "year":             ref.year,
        "month":            ref.month,
        "week_of_year":     ref.isocalendar()[1],
        "days_since_start": days,
        "semester_idx":     sem,
    }


class LgbmPriceService:
    def __init__(self):
        self._models: Dict[tuple, Any] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        for cat in ("notebooks", "all_in_one"):
            for out in ("p25", "p75", "mean"):
                fname = f"{out}_agg_{cat}.pkl"
                path = os.path.join(MODELS_DIR, fname)
                try:
                    with open(path, "rb") as f:
                        self._models[(cat, out)] = pickle.load(f)
                    logging.info(f"[LGBM] Cargado: {fname}")
                except Exception as e:
                    logging.warning(f"[LGBM] No se pudo cargar {fname}: {e}")

    def predict(self, ficha: Dict[str, Any]) -> Optional[Dict]:
        tipo = ficha.get("tipo_equipo")
        cat = TIPO_TO_CAT.get(tipo)
        if cat is None:
            return None

        self._load()
        if not all((cat, o) in self._models for o in ("p25", "p75", "mean")):
            logging.warning(f"[LGBM] Modelos incompletos para {cat}")
            return None

        # GPU
        tiene_gpu = ficha.get("tiene_gpu_dedicada")
        gpu_nombre = ficha.get("gpu_dedicada_nombre")
        if tiene_gpu is False or str(tiene_gpu).lower() == "false":
            gpu_nombre = "Sin GPU dedicada"
        tecnologia_gpu = _derive_gpu_tech(gpu_nombre)

        # Normalizar disco
        disco = ficha.get("tecnologia_disco_principal")
        disco_norm = _DISCO_NORM.get(disco, disco) if disco else None

        row = {
            "total_ram_gb":                  ficha.get("total_ram_gb"),
            "total_almacenamiento_gb":        ficha.get("total_almacenamiento_gb"),
            "nucleos_procesador":             ficha.get("nucleos_procesador"),
            "hilos_procesador":               ficha.get("hilos_procesador"),
            "frecuencia_turbo_procesador_mhz": ficha.get("frecuencia_turbo_procesador_mhz"),
            "frecuencia_ram_mhz":             ficha.get("frecuencia_ram_mhz"),
            "pantalla_pulgadas":              ficha.get("pantalla_pulgadas"),
            "marca":                          ficha.get("marca"),
            "linea_producto":                 None,
            "linea_procesador":               ficha.get("linea_procesador"),
            "generacion_procesador":          ficha.get("generacion_procesador"),
            "tecnologia_ram":                 ficha.get("tecnologia_ram"),
            "tecnologia_disco_principal":     disco_norm,
            "tipo_configuracion_discos":      ficha.get("tipo_configuracion_discos"),
            "gpu_dedicada_nombre":            gpu_nombre,
            "tecnologia_gpu_principal":       tecnologia_gpu,
            "sistema_operativo":              ficha.get("sistema_operativo"),
            "wifi_generacion":                ficha.get("wifi_generacion"),
            "resolucion_pantalla_pixeles":    None,
            **_temporal(date.today()),
        }

        df = pd.DataFrame([row], columns=FEATURE_COLS)

        _NUM_COLS = [
            "total_ram_gb", "total_almacenamiento_gb", "nucleos_procesador",
            "hilos_procesador", "frecuencia_turbo_procesador_mhz",
            "frecuencia_ram_mhz", "pantalla_pulgadas",
            "year", "month", "week_of_year", "days_since_start", "semester_idx",
        ]
        for col in _NUM_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in CAT_COLS:
            df[col] = df[col].astype("category")

        try:
            p25_log = float(self._models[(cat, "p25")].predict(df)[0])
            p75_log = float(self._models[(cat, "p75")].predict(df)[0])
            mean_clp = float(self._models[(cat, "mean")].predict(df)[0])

            return {
                "p25":      int(np.exp(p25_log)),
                "p75":      int(np.exp(p75_log)),
                "mean":     int(mean_clp),
                "category": "Notebooks" if cat == "notebooks" else "All-in-One",
            }
        except Exception as e:
            logging.warning(f"[LGBM] Error en predicción ({cat}): {e}")
            return None
