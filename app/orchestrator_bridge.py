import asyncio
import httpx
import numpy as np
from fastapi import APIRouter, HTTPException
from collections import deque
from typing import Optional, Dict, List

from .auto_predict import auto_forecast
import app.auto as _auto

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

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

# APIs de prédiction ML (sur PC local)
# endpoint disponible : GET /predict?input_data=<valeur_normalisee>
API_RAM_PRED = "http://140.93.1.66:8082/predict"
API_CPU_PRED = "http://140.93.1.66:8083/predict"

# Poids de pondération horizon 7 : p1*7 + p2*6 + ... + p7*1 / 28
HORIZON_WEIGHTS = [7, 6, 5, 4, 3, 2, 1]
HORIZON_SUM     = sum(HORIZON_WEIGHTS)  # 28

router = APIRouter()

# ══════════════════════════════════════════════════════════════
#  BUFFER LATENCE (inchangé)
# ══════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════
#  PRÉDICTION LATENCE (inchangée)
# ══════════════════════════════════════════════════════════════

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
        in_data = np.array(list(buffer), dtype=np.float32)
        pred = auto_forecast(_auto.model, in_data, _auto.selected_horizon, _auto.hyperparameters)
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
    return [round(v * MAX_LATENCY_MS, 2) for v in preds]

# ══════════════════════════════════════════════════════════════
#  MÉTRIQUES WORKERS (/metrics :9100)
# ══════════════════════════════════════════════════════════════

async def fetch_worker_metrics(worker: str) -> Optional[dict]:
    """
    Récupère cpu_used (%), ram_used (%), total_cores, total_ram_gb
    depuis l'agent metrics du worker.
    Retourne None si injoignable.
    """
    ip = WORKER_IPS.get(worker)
    if not ip:
        return None
    url = f"http://{ip}:{WORKER_METRICS_PORT}/metrics"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            result = {
                "cpu_used":     float(data.get("cpu_used", 0)),
                "ram_used":     float(data.get("ram_used", 0)),
                "total_cores":  float(data.get("total_cores", 1)),
                "total_ram_gb": float(data.get("total_ram_gb", 1)),
            }
            print(f"📊 [WORKER METRICS] {worker} → "
                  f"cpu_used={result['cpu_used']}% ram_used={result['ram_used']}% "
                  f"total_cores={result['total_cores']} total_ram_gb={result['total_ram_gb']}GB")
            return result
    except Exception as e:
        print(f"⚠️  [WORKER METRICS] {worker} injoignable : {e}")
        return None

async def fetch_all_workers_metrics() -> Dict[str, Optional[dict]]:
    tasks   = [fetch_worker_metrics(w) for w in WORKERS]
    results = await asyncio.gather(*tasks)
    return {w: results[i] for i, w in enumerate(WORKERS)}

# ══════════════════════════════════════════════════════════════
#  PRÉDICTIONS CPU / RAM via APIs ML
# ══════════════════════════════════════════════════════════════

def _parse_prediction_response(raw) -> Optional[List[float]]:
    """
    Parse la réponse de l'API ML.
    Gère deux formats :
      - liste JSON   : [0.30, 0.31, ...]
      - string numpy : "[0.30837256 0.30407926 ...]"
    Retourne une liste de 7 floats, ou None si échec.
    """
    if isinstance(raw, list):
        return [float(v) for v in raw]
    if isinstance(raw, str):
        # Nettoyer la string numpy : "[0.30837256 0.30407926 ...]"
        cleaned = raw.strip().lstrip("[").rstrip("]")
        parts   = cleaned.split()
        try:
            return [float(p) for p in parts]
        except ValueError:
            pass
    return None

async def fetch_ml_prediction(api_url: str, input_value: float, worker: str, label: str) -> Optional[List[float]]:
    """
    Appelle GET /predict?input_data=<valeur_normalisée>
    L'API n'accepte QUE input_data (pas de paramètre worker).
    input_value est en % (0-100) → on normalise en 0-1 avant envoi.
    Retourne la liste de 7 prédictions normalisées, ou None.
    """
    norm_input = round(input_value / 100.0, 4)
    params = {"input_data": norm_input}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(api_url, params=params)
            resp.raise_for_status()
            data = resp.json()
        print(f"🌐 [ML API] {worker} {label} → GET {api_url}?input_data={norm_input} → {data}")

        raw_pred = data.get("prediction")
        if raw_pred is None:
            print(f"⚠️  [ML API] {worker} {label} : pas de champ 'prediction' dans {data}")
            return None

        preds = _parse_prediction_response(raw_pred)
        if preds is None or len(preds) == 0:
            print(f"⚠️  [ML API] {worker} {label} : parse échoué → {raw_pred}")
            return None

        print(f"🔮 [ML API] {worker} {label} prédictions normalisées : "
              f"{[round(p, 4) for p in preds]}")
        return preds

    except Exception as e:
        print(f"❌ [ML API] {worker} {label} appel échoué ({api_url}) : {e}")
        return None

# ══════════════════════════════════════════════════════════════
#  PONDÉRATION HORIZON (p1*7 + ... + p7*1) / 28
# ══════════════════════════════════════════════════════════════

def _weighted_avg(preds: List[float]) -> float:
    """
    Moyenne pondérée décroissante sur 7 valeurs (ou moins).
    Donne plus de poids aux prédictions proches.
    """
    weights = HORIZON_WEIGHTS[:len(preds)]
    total_w = sum(weights)
    return sum(w * p for w, p in zip(weights, preds)) / total_w

# ══════════════════════════════════════════════════════════════
#  CALCUL DISPONIBILITÉ PRÉDITE
# ══════════════════════════════════════════════════════════════

def compute_availability(
    preds_norm: List[float],
    total_capacity: float,
    label: str,
    worker: str
) -> Optional[float]:
    """
    Calcule la disponibilité prédite en valeur absolue.

    Formule :
      utilisation_predite (%) = weighted_avg(preds_norm) * 100
      disponible            = total_capacity * (1 - utilisation_predite / 100)

    Pour CPU  : total_capacity = total_cores  → résultat en cœurs libres
    Pour RAM  : total_capacity = total_ram_gb → résultat en GB libres
    """
    util_pct  = _weighted_avg(preds_norm) * 100.0
    available = total_capacity * (1.0 - util_pct / 100.0)
    available = round(available, 3)
    print(f"📐 [DISPO] {worker} {label} : "
          f"util_predite={util_pct:.1f}% → dispo={available:.3f} "
          f"(capacité={total_capacity})")
    return available

# ══════════════════════════════════════════════════════════════
#  PUSH VERS ORCHESTRATEUR (inchangés)
# ══════════════════════════════════════════════════════════════

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
    """
    Pousse vers l'orchestrateur les disponibilités prédites (en valeur absolue).

    cpu_map[worker] = liste de 7 prédictions normalisées CPU
    ram_map[worker] = liste de 7 prédictions normalisées RAM

    L'orchestrateur reçoit ces listes et calculera lui-même la pondération
    via compute_cpu_score / compute_ram_score (comme avant).
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{orchestrator_url}/predictions_cpu_ram",
                json={"cpu_predictions": cpu_map, "ram_predictions": ram_map}
            )
            print(f"📤 [BRIDGE] Prédictions CPU/RAM (ML réelles) poussées → {resp.status_code}")
    except Exception as e:
        print(f"❌ [BRIDGE] Push CPU/RAM échoué : {e}")

# ══════════════════════════════════════════════════════════════
#  PUSH DISPONIBILITÉS (nouveau endpoint orchestrateur)
# ══════════════════════════════════════════════════════════════

async def _push_worker_availability(orchestrator_url: str, availability_map: dict):
    """
    Pousse les disponibilités prédites calculées par le bridge vers l'orchestrateur.

    Format :
    {
      "space_1_edge": {
          "cpu_dispo_predite": 2.3,   # cœurs libres prédits
          "ram_dispo_predite": 3.1,   # GB libres prédits
          "cpu_dispo_actuel":  1.8,   # cœurs libres actuels
          "ram_dispo_actuel":  2.9,   # GB libres actuels
          "total_cores":       4.0,
          "total_ram_gb":      8.0,
      },
      ...
    }
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{orchestrator_url}/worker_availability",
                json={"availability": availability_map}
            )
            print(f"📤 [BRIDGE] Disponibilités workers poussées → {resp.status_code}")
    except Exception as e:
        print(f"❌ [BRIDGE] Push disponibilités échoué : {e}")

# ══════════════════════════════════════════════════════════════
#  VÉRIFIER MODE INTENT
# ══════════════════════════════════════════════════════════════

async def _intent_is_active(orchestrator_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp  = await client.get(f"{orchestrator_url}/status")
            return resp.json().get("mode") == "enhanced"
    except:
        return False

# ══════════════════════════════════════════════════════════════
#  ENDPOINT PRINCIPAL : POST /latency
# ══════════════════════════════════════════════════════════════

@router.post("/latency")
async def receive_from_picar(
    data: dict,
    orchestrator_url: str = ORCHESTRATOR_DEFAULT
):
    """
    Reçoit les latences de la PiCar, calcule les prédictions
    et pousse vers l'orchestrateur.

    Mode CLASSIC  : latences réelles + prédictions latence uniquement.
    Mode ENHANCED : + métriques workers réelles + prédictions CPU/RAM via ML
                    + calcul disponibilités prédites.
    """
    latencies = data.get("latencies", {})
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

    asyncio.ensure_future(_push_latencies(orchestrator_url, latencies))
    if lat_map:
        asyncio.ensure_future(_push_predictions(orchestrator_url, lat_map))
    else:
        print("⏳ [BRIDGE] Buffer latence en remplissage...")

    # ── 2. Mode ENHANCED : CPU/RAM via APIs ML ───────────────────
    intent_active = await _intent_is_active(orchestrator_url)

    if not intent_active:
        print("🔵 [BRIDGE] Mode CLASSIC → pas de prédictions CPU/RAM")
        return {
            "status":            "ok",
            "workers_received":  list(latencies.keys()),
            "workers_predicted": list(lat_map.keys()),
            "intent_active":     False,
            "mode":              "classic",
        }

    print("🎯 [BRIDGE] Mode ENHANCED → collecte métriques + prédictions ML CPU/RAM")

    # 2a. Récupérer métriques réelles de tous les workers
    all_metrics = await fetch_all_workers_metrics()

    # 2b. Pour chaque worker : appeler les APIs ML en parallèle
    cpu_pred_tasks = []
    ram_pred_tasks = []
    valid_workers  = []

    for worker in WORKERS:
        metrics = all_metrics.get(worker)
        if metrics is None:
            print(f"⚠️  [BRIDGE] {worker} : métriques indisponibles → ignoré pour ENHANCED")
            continue
        valid_workers.append(worker)
        cpu_pred_tasks.append(
            fetch_ml_prediction(API_CPU_PRED, metrics["cpu_used"], worker, "CPU")
        )
        ram_pred_tasks.append(
            fetch_ml_prediction(API_RAM_PRED, metrics["ram_used"], worker, "RAM")
        )

    if not valid_workers:
        print("⚠️  [BRIDGE] Aucun worker joignable → pas de push CPU/RAM")
        return {
            "status": "ok", "workers_received": list(latencies.keys()),
            "workers_predicted": list(lat_map.keys()), "intent_active": True,
            "mode": "enhanced_fallback_no_workers",
        }

    # Lancer tous les appels ML en parallèle
    cpu_results = await asyncio.gather(*cpu_pred_tasks)
    ram_results = await asyncio.gather(*ram_pred_tasks)

    # 2c. Construire les maps de prédictions et de disponibilités
    cpu_preds_map    = {}   # worker → liste 7 preds normalisées (pour orchestrateur)
    ram_preds_map    = {}
    availability_map = {}   # worker → {cpu_dispo_predite, ram_dispo_predite, ...}

    for i, worker in enumerate(valid_workers):
        metrics   = all_metrics[worker]
        cpu_preds = cpu_results[i]
        ram_preds = ram_results[i]

        total_cores  = metrics["total_cores"]
        total_ram_gb = metrics["total_ram_gb"]
        cpu_used     = metrics["cpu_used"]
        ram_used     = metrics["ram_used"]

        # Disponibilités actuelles (sans prédiction)
        cpu_dispo_actuel = round(total_cores  * (1.0 - cpu_used / 100.0), 3)
        ram_dispo_actuel = round(total_ram_gb * (1.0 - ram_used / 100.0), 3)

        # Disponibilités prédites (avec pondération ML)
        cpu_dispo_predite = None
        ram_dispo_predite = None

        if cpu_preds and len(cpu_preds) > 0:
            cpu_preds_map[worker]  = cpu_preds
            cpu_dispo_predite      = compute_availability(
                cpu_preds, total_cores, "CPU", worker
            )
        else:
            print(f"⚠️  [BRIDGE] {worker} CPU : prédiction ML échouée → utilisation valeur actuelle")
            cpu_dispo_predite = cpu_dispo_actuel

        if ram_preds and len(ram_preds) > 0:
            ram_preds_map[worker]  = ram_preds
            ram_dispo_predite      = compute_availability(
                ram_preds, total_ram_gb, "RAM", worker
            )
        else:
            print(f"⚠️  [BRIDGE] {worker} RAM : prédiction ML échouée → utilisation valeur actuelle")
            ram_dispo_predite = ram_dispo_actuel

        availability_map[worker] = {
            "cpu_dispo_predite": cpu_dispo_predite,
            "ram_dispo_predite": ram_dispo_predite,
            "cpu_dispo_actuel":  cpu_dispo_actuel,
            "ram_dispo_actuel":  ram_dispo_actuel,
            "total_cores":       total_cores,
            "total_ram_gb":      total_ram_gb,
        }

        print(f"✅ [BRIDGE] {worker} → "
              f"cpu_dispo_actuel={cpu_dispo_actuel:.2f} cores / "
              f"cpu_dispo_predite={cpu_dispo_predite:.2f} cores | "
              f"ram_dispo_actuel={ram_dispo_actuel:.2f} GB / "
              f"ram_dispo_predite={ram_dispo_predite:.2f} GB")

    # 2d. Pousser vers orchestrateur
    if cpu_preds_map or ram_preds_map:
        asyncio.ensure_future(
            _push_cpu_ram_predictions(orchestrator_url, cpu_preds_map, ram_preds_map)
        )

    if availability_map:
        asyncio.ensure_future(
            _push_worker_availability(orchestrator_url, availability_map)
        )

    return {
        "status":               "ok",
        "workers_received":     list(latencies.keys()),
        "workers_predicted_lat": list(lat_map.keys()),
        "workers_enhanced":     list(availability_map.keys()),
        "intent_active":        True,
        "mode":                 "enhanced",
        "availability_summary": {
            w: {
                "cpu_dispo_predite": v["cpu_dispo_predite"],
                "ram_dispo_predite": v["ram_dispo_predite"],
            }
            for w, v in availability_map.items()
        }
    }

# ══════════════════════════════════════════════════════════════
#  ENDPOINTS DE DEBUG (inchangés)
# ══════════════════════════════════════════════════════════════

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
