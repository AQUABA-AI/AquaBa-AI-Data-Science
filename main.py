from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Data Science API running"}

@app.get("/demand")
def demand():
    return {"message": "Demand forecast coming soon"}

@app.get("/spoilage")
def spoilage():
    return {"message": "Spoilage alerts coming soon"}
