from fastapi import FastAPI
from app.system import SustainableFarmingSystem
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(
    title="Sustainable Farming System",
    description="Simple API for farming recommendations",
    version="1.0.0"
)

# Initialize the system
system = SustainableFarmingSystem()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/load-sample-data")
async def load_sample_data():
    system.load_sample_data()
    return {"message": "Sample data loaded successfully"}

@app.post("/recommendations/{farm_id}/{market_id}")
async def generate_recommendations(farm_id: int, market_id: int):
    recommendation = system.generate_recommendations(farm_id, market_id)
    return recommendation

@app.post("/farm-data")
async def add_farm_data(data: dict):
    system.db.insert_farm_data(data)
    return {"message": "Farm data added"}

@app.post("/market-data")
async def add_market_data(data: dict):
    system.db.insert_market_data(data)
    return {"message": "Market data added"}

# @app.on_event("shutdown")
# async def shutdown_event():
#     system.db.close()