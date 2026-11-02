# A model as a web service (FastAPI): input validation, model version in the response, request log
import joblib, time
from fastapi import FastAPI
from pydantic import BaseModel, Field

bundle = joblib.load("model_v1.joblib")
model, meta = bundle["model"], bundle["meta"]
app = FastAPI(title="APU-Expert: fuel fault model")
LOG = []                                              # in a real system, a database or a queue

class Params(BaseModel):                              # input schema with the limits of admissible values
    egt: float = Field(ge=300, le=900, description="gas temperature, °C")
    start_time: float = Field(ge=0, le=300, description="start time, s")
    fuel_flow: float = Field(ge=0, le=400, description="fuel flow, kg/h")
    vibration: float = Field(ge=0, le=50, description="vibration, mm/s")

@app.get("/health")
def health():
    return {"status": "ok", "model": meta["name"], "version": meta["version"]}

@app.post("/predict")
def predict(p: Params):
    x = [[p.egt, p.start_time, p.fuel_flow, p.vibration]]
    prob = float(model.predict_proba(x)[0, 1])
    LOG.append({"t": time.time(), "input": p.model_dump(), "prob": prob, "version": meta["version"]})
    return {"fuel_fault_probability": round(prob, 3), "model_version": meta["version"]}
