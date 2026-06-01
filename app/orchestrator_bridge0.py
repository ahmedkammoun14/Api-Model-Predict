import asyncio
import httpx
from fastapi import APIRouter, HTTPException
from collections import deque
from typing import Optional

from .auto_configure import find_best_model
from .auto_predict   import auto_forecast
import app.auto as _auto

# ── Configuration ──────────────────────────────────────────────
MAX_LATENCY_MS       = 300.0
ORCHESTRATOR_DEFAULT = "http://194.199.113.8:6000"
WORKERS = ["space_1_edge", "space_2_edge", "space_3_edge", "space_4_edge"]

router = APIRouter()

# ── Buffers look_back par worker ───────────────────────────────
look_back_buffers: dict[str, deque] = {}


def _get_look_back() -> int:
    if _auto.hyperparameters:
        return _auto.hyperparameters.get(
            "look_back",
            _auto.hyperparameters.get("window_length", 11)
        )
    return 11


def _ensure_buffers():
    lb = _get_look_back()
    for w in WORKERS:
        if w not in look_back_buffers:
            look_back_buffers[w] = deque(maxlen=lb)


# ── Push prédictions vers l'orchestrateur ─────────────────────
async def _push_predictions(orchestrator_url: str, predictions_map: dict):
    payload = {"predictions": predictions_map}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{orchestrator_url}/predictions", json=payload
            )
            print(f"[BRIDGE PUSH PRED] → {resp.status_code} | workers={list(predictions_map.keys())}")
    except Exception as e:
        print(f"[BRIDGE PUSH PRED] ❌ : {e}")


# ── Push latences vers l'orchestrateur ────────────────────────
async def _push_latencies(orchestrator_url: str, latencies: dict):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{orchestrator_url}/latency", json={"latencies": latencies}
            )
            print(f"[BRIDGE PUSH LAT]  → {resp.status_code} | {latencies}")
    except Exception as e:
        print(f"[BRIDGE PUSH LAT]  ❌ : {e}")


def _compute_prediction(worker: str, normalized: float) -> Optional[list]:
    _ensure_buffers()
    look_back_buffers[worker].append(normalized)

    lb = _get_look_back()
    if len(look_back_buffers[worker]) < lb:
        print(f"[BRIDGE] {worker} buffer {len(look_back_buffers[worker])}/{lb} — pas encore prêt")
        return None

    if _auto.model is None or _auto.processed_dataset is None:
        print(f"[BRIDGE] Modèle non initialisé pour {worker}")
        return None

    try:
        import numpy as np

        # ← FIX : tableau 1D — auto_forecast fait lui-même le reshape
        # LSTM : np.reshape(input_data, (1, 1, look_back))
        # ESN  : np.reshape(input_data, (1, look_back))
        in_data = np.array(list(look_back_buffers[worker]), dtype=np.float32)
        # shape attendu : (look_back,)  ex: (37,)

        pred = auto_forecast(
            _auto.model,
            in_data,
            _auto.selected_horizon,
            _auto.hyperparameters
        )

        predictions = [round(float(v) * MAX_LATENCY_MS, 2) for v in pred[0]]
        print(f"[BRIDGE PRED] {worker} input={normalized:.4f} → {predictions}")
        return predictions

    except Exception as e:
        print(f"[BRIDGE PRED] ❌ {worker} : {e}")
        return None


# ─────────────────────────────────────────────────────────────
# POST /latency
# ─────────────────────────────────────────────────────────────
@router.post("/latency")
async def receive_latency_from_picar(
    data: dict,
    orchestrator_url: str = ORCHESTRATOR_DEFAULT
):
    latencies: dict = data.get("latencies", {})
    if not latencies:
        raise HTTPException(status_code=400, detail="latencies manquantes")

    predictions_map = {}

    for worker, lat_ms in latencies.items():
        if lat_ms == -1 or worker not in WORKERS:
            continue

        normalized = round(float(lat_ms) / MAX_LATENCY_MS, 4)
        preds = _compute_prediction(worker, normalized)

        if preds:
            predictions_map[worker] = preds

    asyncio.ensure_future(_push_latencies(orchestrator_url, latencies))

    if predictions_map:
        asyncio.ensure_future(_push_predictions(orchestrator_url, predictions_map))
    else:
        print("[BRIDGE] Pas de prédictions à pousser (buffers en cours de remplissage)")

    return {
        "status":            "ok",
        "workers_received":  list(latencies.keys()),
        "workers_predicted": list(predictions_map.keys()),
    }


# ─────────────────────────────────────────────────────────────
# GET /predict_and_push  (debug)
# ─────────────────────────────────────────────────────────────
@router.get("/predict_and_push")
async def predict_and_push(
    input_data:       float,
    worker:           str,
    orchestrator_url: Optional[str] = ORCHESTRATOR_DEFAULT,
):
    if _auto.model is None or _auto.processed_dataset is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non initialisé — lancer POST /main d'abord"
        )

    normalized = round(input_data / MAX_LATENCY_MS, 4)
    preds = _compute_prediction(worker, normalized)

    if preds is None:
        return {"worker": worker, "prediction": [], "status": "buffer_filling"}

    asyncio.ensure_future(_push_predictions(orchestrator_url, {worker: preds}))

    return {
        "worker":     worker,
        "prediction": preds,
        "pushed_to":  orchestrator_url,
    }


# ─────────────────────────────────────────────────────────────
# POST /intent
# ─────────────────────────────────────────────────────────────
INTENT_ENGINE_URL = "http://127.0.0.1:7001/intent"

@router.post("/intent")
async def relay_intent(data: dict, orchestrator_url: str = ORCHESTRATOR_DEFAULT):
    intention = data.get("intention", "").strip()
    if not intention:
        raise HTTPException(status_code=400, detail="intention vide")

    try:
        async with httpx.AsyncClient(timeout=130.0) as client:
            resp = await client.post(
                INTENT_ENGINE_URL,
                json={"intention": intention}
            )
            resp.raise_for_status()
            slos_payload = resp.json()
            print(f"[BRIDGE INTENT] SLOs reçus: {slos_payload}")
    except Exception as e:
        print(f"[BRIDGE INTENT] ❌ Intent engine: {e}")
        raise HTTPException(status_code=503, detail=f"Intent engine indisponible: {e}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{orchestrator_url}/intent",
                json=slos_payload 
            )
            print(f"[BRIDGE INTENT] → orchestrateur {resp.status_code}")
            return {"status": "ok", "slos": slos_payload}
    except Exception as e:
        print(f"[BRIDGE INTENT] ❌ Orchestrateur: {e}")
        raise HTTPException(status_code=503, detail=f"Orchestrateur indisponible: {e}") 
