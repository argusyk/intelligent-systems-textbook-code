# Модель як вебсервіс (FastAPI): перевірка входу, версія моделі у відповіді, журнал запитів
import joblib, time
from fastapi import FastAPI
from pydantic import BaseModel, Field

bundle = joblib.load("model_v1.joblib")
model, meta = bundle["model"], bundle["meta"]
app = FastAPI(title="ДСУ-Експерт: модель паливних несправностей")
LOG = []                                              # у реальній системі – база даних або черга

class Params(BaseModel):                              # схема входу з межами допустимих значень
    egt: float = Field(ge=300, le=900, description="температура газів, °C")
    start_time: float = Field(ge=0, le=300, description="час запуску, с")
    fuel_flow: float = Field(ge=0, le=400, description="витрата палива, кг/год")
    vibration: float = Field(ge=0, le=50, description="вібрація, мм/с")

@app.get("/health")
def health():
    return {"status": "ok", "model": meta["name"], "version": meta["version"]}

@app.post("/predict")
def predict(p: Params):
    x = [[p.egt, p.start_time, p.fuel_flow, p.vibration]]
    prob = float(model.predict_proba(x)[0, 1])
    LOG.append({"t": time.time(), "input": p.model_dump(), "prob": prob, "version": meta["version"]})
    return {"fuel_fault_probability": round(prob, 3), "model_version": meta["version"]}
