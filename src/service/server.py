"""
FastAPI application exposing the course-required endpoints.

Contract:
  GET /recommend/{userid}  -> returns a single line of up to 20 movie IDs
                              as comma-separated values (text/plain).
  GET /healthz             -> simple JSON health check for ops/monitoring.
"""

from fastapi import FastAPI, Response
from .ranking import recommend_for_user, healthcheck

# Create the web app object once at import time.
# Uvicorn/ASGI will reuse this app for every request.
app = FastAPI(title="Home Alone Recommendation Service")


@app.get("/recommend/{userid}")
def recommend(userid: int):
    """
    Handles recommendation requests.

    Args:
        userid (int): Numeric user identifier from the URL path.

    Returns:
        Plain-text HTTP response with <=20 comma-separated movie IDs.
        IMPORTANT: The course infrastructure expects *text/plain*,
        not JSON, and the body must be exactly one line.
    """
    # Delegate to our ranking module. This keeps the HTTP layer thin.
    ids = recommend_for_user(userid, k=20)

    # Join with commas and explicitly set text/plain to satisfy the spec.
    return Response(",".join(ids), media_type="text/plain; charset=utf-8")


@app.get("/healthz")
def healthz():
    """
    Liveness/readiness probe.

    Returns:
        JSON like {"ok": true, "msg": "bootstrap"}.
        This endpoint is safe for load balancers and monitoring systems.
    """
    ok, msg = healthcheck()
    return {"ok": ok, "msg": msg}
