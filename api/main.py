"""genphish-guard REST API + web console.

Run:
    uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000  — the web console is served from the same
origin, so there is no CORS setup and no second process to start.

Interactive API docs: http://localhost:8000/docs
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .service import service

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

app = FastAPI(
    title="genphish-guard API",
    version="2.0.0",
    description=(
        "Generator-agnostic detection of LLM-generated phishing without "
        "labelled target data, with confidence-graded zero-trust response.\n\n"
        "M.Tech Cybersecurity dissertation — KIIT University / LTIMindtree."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ schemas
class DetectRequest(BaseModel):
    text: str = Field(..., description="Raw email body (headers optional).",
                      min_length=1)
    detector: str = Field("xgboost", description="xgboost | logreg | rf | mlp")
    trained_on: str = Field("pooled",
                            description="generator name, or 'pooled'")
    calibration: str = Field("stale",
                             description="stale | dm | em | selftrain | oracle")
    t_lo: float | None = Field(None, description="DELIVER/DEFUSE boundary")
    t_hi: float | None = Field(None, description="DEFUSE/BLOCK boundary")
    raise_incident: bool = Field(True,
                                 description="Create a SOAR incident on BLOCK/DEFUSE")


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)
    detector: str = "xgboost"
    trained_on: str = "pooled"
    calibration: str = "stale"
    raise_incidents: bool = False


class DriftStepRequest(BaseModel):
    session_id: str
    score: float = Field(..., ge=0.0, le=1.0)


class DispatchRequest(BaseModel):
    incident_id: str
    endpoint: str | None = Field(
        None, description="Webhook URL. Omit for simulation mode.")


def _guard() -> None:
    if not service.ready:
        raise HTTPException(503, f"model not ready: {service.boot_error}")


# ------------------------------------------------------------------ system
@app.get("/api/health", tags=["system"])
def health():
    return {
        "status": "ok" if service.ready else "degraded",
        "ready": service.ready,
        "error": service.boot_error,
        "data_mode": service.data_mode,
        "detectors": {k: list(v.keys()) for k, v in service.bundles.items()},
        "version": app.version,
    }


@app.get("/api/stats", tags=["system"])
def stats():
    _guard()
    return service.stats()


@app.get("/api/samples", tags=["system"])
def samples():
    return {"samples": service.samples()}


# ------------------------------------------------------------------ detect
@app.post("/api/detect", tags=["detection"])
def detect(req: DetectRequest):
    _guard()
    try:
        return service.detect(
            req.text, req.detector, req.trained_on, req.calibration,
            req.t_lo, req.t_hi, req.raise_incident,
        )
    except KeyError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/detect/batch", tags=["detection"])
def detect_batch(req: BatchRequest):
    _guard()
    try:
        return service.detect_batch(
            req.texts, req.detector, req.trained_on, req.calibration,
            req.raise_incidents,
        )
    except KeyError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/features", tags=["detection"])
def features(text: str = Query(..., min_length=1)):
    _guard()
    return {"features": service.extract(text)}


# ------------------------------------------------------------- experiments
@app.get("/api/transfer", tags=["experiments"])
def transfer():
    _guard()
    return service.transfer()


@app.get("/api/calibration", tags=["experiments"])
def calibration():
    _guard()
    return service.calibration_table()


# ------------------------------------------------------------------- drift
@app.get("/api/drift/precomputed", tags=["drift"])
def drift_precomputed():
    return service.drift_precomputed()


@app.post("/api/drift/session", tags=["drift"])
def drift_session(band_lo: float = 0.35, band_hi: float = 0.65,
                  warmup: int = 30):
    _guard()
    return {"session_id": service.drift_new_session(band_lo, band_hi, warmup)}


@app.post("/api/drift/step", tags=["drift"])
def drift_step(req: DriftStepRequest):
    _guard()
    try:
        return service.drift_step(req.session_id, req.score)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/drift/simulate", tags=["drift"])
def drift_simulate(n: int = Query(400, ge=50, le=3000),
                   shift_at: int = Query(200, ge=10),
                   detector: str = "xgboost",
                   source: str | None = None,
                   target: str | None = None):
    _guard()
    try:
        return service.drift_simulate(n, shift_at, detector, source, target)
    except KeyError as exc:
        raise HTTPException(400, str(exc))


# -------------------------------------------------------------------- soar
@app.get("/api/soar/incidents", tags=["soar"])
def soar_incidents(limit: int = Query(100, ge=1, le=500)):
    return service.soar_log(limit)


@app.delete("/api/soar/incidents", tags=["soar"])
def soar_clear():
    return service.soar_clear()


@app.post("/api/soar/dispatch", tags=["soar"])
def soar_dispatch(req: DispatchRequest):
    try:
        return service.soar_dispatch(req.incident_id, req.endpoint)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/soar/playbook/{tier}", tags=["soar"])
def soar_playbook(tier: str):
    from genphish import soar as SOAR
    t = tier.upper()
    if t not in ("BLOCK", "DEFUSE", "DELIVER"):
        raise HTTPException(400, "tier must be BLOCK, DEFUSE or DELIVER")
    stub = SOAR.Incident(id="PREVIEW", timestamp=0.0, tier=t, confidence=0.0,
                         threshold=0.0, cal_method="n/a", source_det="n/a",
                         severity="n/a")
    return {"tier": t, "steps": SOAR.playbook_response(stub)}


# --------------------------------------------------------------- web console
if WEB.exists():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        f = WEB / "index.html"
        if f.exists():
            return FileResponse(str(f))
        return JSONResponse({"detail": "web console not built"}, 404)

    @app.get("/model.json", include_in_schema=False)
    def model_json():
        f = WEB / "model.json"
        if f.exists():
            return FileResponse(str(f))
        return JSONResponse({"detail": "model.json not built"}, 404)
