# Automated tests of the service: health, correct response, rejection of invalid input
from fastapi.testclient import TestClient
from service import app

c = TestClient(app)
r = c.get("/health"); print("health:", r.status_code, r.json())
r = c.post("/predict", json={"egt": 705, "start_time": 52, "fuel_flow": 155, "vibration": 3.1})
print("faulty unit:", r.status_code, r.json())
r = c.post("/predict", json={"egt": 600, "start_time": 33, "fuel_flow": 115, "vibration": 2.8})
print("healthy unit:", r.status_code, r.json())
r = c.post("/predict", json={"egt": 7050, "start_time": 52, "fuel_flow": 155, "vibration": 3.1})
print("input error EGT = 7050:", r.status_code, r.json()["detail"][0]["msg"])
