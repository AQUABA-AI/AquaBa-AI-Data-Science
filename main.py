from fastapi import FastAPI
from demand_forecast import predict_demand
from spoilage_alerts import check_spoilage

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Data Science API running"}

@app.get("/demand")
def demand():
    return predict_demand()

@app.get("/spoilage")
def spoilage():
    return check_spoilage()
