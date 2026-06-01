import asyncio
import httpx
from fastapi import APIRouter, HTTPException
from collections import deque
from typing import Optional, Dict

from .auto_predict import auto_forecast
import app.auto as _auto

# ── Configuration ──────────────────────────────────────────────
MAX_LATENCY_MS       = 300.0
ORCHESTRATOR_DEFAULT = "http://194.199.113.8:6000"
WORKERS = ["space_1_edge", "space_2_edge", "space_3_edge", "space_4_edge"]
WORKER_IPS = {
    "space_1_edge": "194.199.113.18",
    "space_2_edge": "194.199.113.28",
    "space_3_edge": "194.199.113.66",
    "space_4_edge": "194.199.113.69",
}
WORKER_METRICS_PORT = 9100

router = APIRouter()

# ── Buffers look_back par worker pour la latence seulement ─────
look_back_buffers_lat: Dict[str, deque] = {}

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
        if w not in look_back_buffers_lat:
            look_back_buffers_lat[w] = deque(maxlen=lb)

# ── Prédiction latence (inchangée, fonctionnelle) ─────────────
def _predict_from_buffer(buffer: deque, norm_value: float, worker: str, label: str) -> Optional[list]:
    buffer.append(norm_value)
    lb = _get_look_back()
    if len(buffer) < lb:
        print(f"⏳ [BRIDGE] {worker} {label} buffer {len(buffer)}/{lb} — remplissage...")
        return None
    if _auto.model is None or _auto.processed_dataset is None:
        print(f"❌ [BRIDGE] Modèle non initialisé pour {label}")
        return None
    try:
        import numpy as np
        in_data = np.array(list(buffer), dtype=np.float32)
        # auto_forecast retourne un tableau de shape (1, horizon)
        pred = auto_forecast(_auto.model, in_data, _auto.selected_horizon, _auto.hyperparameters)
        # Conversion en liste 1D
        return [round(float(v), 4) for v in pred[0]]
    except Exception as e:
        print(f"❌ [BRIDGE PRED] {worker} {label} : {e}")
        return None

def _compute_lat_prediction(worker: str, lat_ms: float) -> Optional[list]:
    _ensure_buffers()
    preds = _predict_from_buffer(
        look_back_buffers_lat[worker],
        round(lat_ms / MAX_LATENCY_MS, 4),
        worker, "LAT"
    )
    if preds is None:
        return None
    return [round(v * MAX_LATENCY_MS, 2) for v in preds]  # → ms

# ── Collecte des métriques réelles des workers ─────────────────
async def fetch_worker_metrics(worker: str) -> tuple[Optional[float], Optional[float]]:
    ip = WORKER_IPS.get(worker)
    if not ip:
        return None, None
    url = f"http://{ip}:{WORKER_METRICS_PORT}/metrics"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            cpu_used = data.get("cpu_used")
            ram_used = data.get("ram_used")
            if cpu_used is not None and ram_used is not None:
                print(f"📊 [WORKER] {worker} → CPU réel={cpu_used}% RAM réel={ram_used}%")
                return float(cpu_used), float(ram_used)
    except Exception as e:
        print(f"⚠️ [WORKER] {worker} injoignable : {e}")
    return None, None

async def fetch_all_workers_metrics() -> Dict[str, tuple[Optional[float], Optional[float]]]:
    tasks = [fetch_worker_metrics(w) for w in WORKERS]
    results = await asyncio.gather(*tasks)
    return {w: results[i] for i, w in enumerate(WORKERS)}

# ── Push vers orchestrateur (inchangé) ─────────────────────────
async def _push_predictions(orchestrator_url: str, predictions_map: dict):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{orchestrator_url}/predictions",
                json={"predictions": predictions_map}
            )
            print(f"📤 [BRIDGE] Prédictions LATENCE poussées → {resp.status_code}")
    except Exception as e:
        print(f"❌ [BRIDGE] Push latence échoué : {e}")

async def _push_latencies(orchestrator_url: str, latencies: dict):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{orchestrator_url}/latency",
                json={"latencies": latencies}
            )
            print(f"📤 [BRIDGE] Latences réelles poussées → {resp.status_code}")
    except Exception as e:
        print(f"❌ [BRIDGE] Push latences échoué : {e}")

async def _push_cpu_ram_predictions(orchestrator_url: str, cpu_map: dict, ram_map: dict):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{orchestrator_url}/predictions_cpu_ram",
                json={"cpu_predictions": cpu_map, "ram_predictions": ram_map}
            )
            print(f"📤 [BRIDGE] Prédictions CPU/RAM poussées → {resp.status_code}")
    except Exception as e:
        print(f"❌ [BRIDGE] Push CPU/RAM échoué : {e}")

# ── Vérifier si intent actif ───────────────────────────────────
async def _intent_is_active(orchestrator_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{orchestrator_url}/status")
            return resp.json().get("mode") == "enhanced"
    except:
        return False

# ─────────────────────────────────────────────────────────────
# POST /latency — endpoint principal MODIFIÉ
# ─────────────────────────────────────────────────────────────
@router.post("/latency")
async def receive_from_picar(
    data: dict,
    orchestrator_url: str = ORCHESTRATOR_DEFAULT
):
    latencies   = data.get("latencies", {})
    # On ignore cpu_percent/ram_percent de la PiCar pour les workers

    if not latencies:
        raise HTTPException(status_code=400, detail="latencies manquantes")

    # ── 1. Prédictions latence (inchangées) ─────────────────────
    lat_map = {}
    for worker, lat_ms in latencies.items():
        if lat_ms == -1 or worker not in WORKERS:
            continue
        preds = _compute_lat_prediction(worker, float(lat_ms))
        if preds:
            lat_map[worker] = preds

    # Envoi des latences réelles
    asyncio.ensure_future(_push_latencies(orchestrator_url, latencies))
    if lat_map:
        asyncio.ensure_future(_push_predictions(orchestrator_url, lat_map))
    else:
        print("⏳ [BRIDGE] Buffer latence en remplissage...")

    # ── 2. CPU/RAM : récupération des métriques réelles des workers ──
    intent_active = await _intent_is_active(orchestrator_url)

    if intent_active:
        print("🎯 [BRIDGE] Mode INTENT actif → collecte des métriques réelles des workers")
        workers_metrics = await fetch_all_workers_metrics()

        cpu_map = {}
        ram_map = {}
        for worker in WORKERS:
            cpu_real, ram_real = workers_metrics.get(worker, (None, None))
            if cpu_real is None or ram_real is None:
                print(f"⚠️ [BRIDGE] {worker} : métriques non disponibles → pas de prédictions CPU/RAM")
                continue

            # Construction de prédictions simplifiées : 7 fois la valeur actuelle
            cpu_preds = [cpu_real] * 7
            ram_preds = [ram_real] * 7
            cpu_map[worker] = cpu_preds
            ram_map[worker] = ram_preds
            print(f"🔮 [BRIDGE] {worker} → prédictions CPU (répétées) : {cpu_preds}")
            print(f"🔮 [BRIDGE] {worker} → prédictions RAM (répétées) : {ram_preds}")

        if cpu_map and ram_map:
            asyncio.ensure_future(_push_cpu_ram_predictions(orchestrator_url, cpu_map, ram_map))
        else:
            print("⚠️ [BRIDGE] Aucune prédiction CPU/RAM générée (workers injoignables)")
    else:
        print("🔵 [BRIDGE] Mode CLASSIC → pas de prédictions CPU/RAM")

    return {
        "status":            "ok",
        "workers_received":  list(latencies.keys()),
        "workers_predicted": list(lat_map.keys()),
        "intent_active":     intent_active,
    }

# ── Endpoints de debug et relais intent (inchangés) ────────────
@router.get("/predict_and_push")
async def predict_and_push(
    input_data:       float,
    worker:           str,
    orchestrator_url: Optional[str] = ORCHESTRATOR_DEFAULT,
):
    if _auto.model is None or _auto.processed_dataset is None:
        raise HTTPException(status_code=503, detail="Modèle non initialisé")
    preds = _compute_lat_prediction(worker, input_data)
    if preds is None:
        return {"worker": worker, "prediction": [], "status": "buffer_filling"}
    asyncio.ensure_future(_push_predictions(orchestrator_url, {worker: preds}))
    return {"worker": worker, "prediction": preds, "pushed_to": orchestrator_url}

INTENT_ENGINE_URL = "http://127.0.0.1:7001/intent"

@router.post("/intent")
async def relay_intent(data: dict, orchestrator_url: str = ORCHESTRATOR_DEFAULT):
    intention = data.get("intention", "").strip()
    if not intention:
        raise HTTPException(status_code=400, detail="intention vide")
    try:
        async with httpx.AsyncClient(timeout=130.0) as client:
            resp = await client.post(INTENT_ENGINE_URL, json={"intention": intention})
            resp.raise_for_status()
            slos_payload = resp.json()
            print(f"🎯 [INTENT] SLOs reçus: {slos_payload}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Intent engine indisponible: {e}")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{orchestrator_url}/intent",
                json=slos_payload
            )
            print(f"📡 [INTENT] Transmis à l'orchestrateur → {resp.status_code}")
            return {"status": "ok", "slos": slos_payload}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Orchestrateur indisponible: {e}")
