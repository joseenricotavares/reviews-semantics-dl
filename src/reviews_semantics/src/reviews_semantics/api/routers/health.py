from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe - never depends on the model being loaded."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request, response: Response) -> dict[str, str]:
    """Readiness probe - 503 until the model has finished loading."""
    if getattr(request.app.state, "predictor", None) is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "loading"}
    return {"status": "ready"}
