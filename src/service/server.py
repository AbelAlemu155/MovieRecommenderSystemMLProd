from fastapi import FastAPI, Response
from service.ranking import recommend_for_user, healthcheck

app = FastAPI(title="Recommendation Service")

@app.get("/recommend/{userid}")
def recommend(userid: int):
    ids = recommend_for_user(userid, k=10)
    return Response(",".join(ids), media_type="text/plain; charset=utf-8")

@app.get("/healthz")
def healthz():
    ok, msg = healthcheck()
    return {"ok": ok, "msg": msg}
