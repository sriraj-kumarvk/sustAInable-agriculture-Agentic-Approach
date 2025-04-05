from .database import Database
from .farmer_advisor import FarmerAdvisorAgent
from .market_researcher import MarketResearcherAgent
from .coordinator import CoordinatorAgent

class SustainableFarmingSystem:
    """Main system that coordinates the multi-agent framework."""
    
    def __init__(self):
        self.db = Database()
        self.farmer_advisor = FarmerAdvisorAgent(self.db)
        self.market_researcher = MarketResearcherAgent(self.db)
        self.coordinator = CoordinatorAgent(self.db, self.farmer_advisor, self.market_researcher)
    
    def load_sample_data(self):
        """Load sample data for demonstration purposes."""
        # Sample farm data
        farm_data = [
            {
                'Farm_ID': 10002,
                'Soil_pH': 6.5,
                'Soil_Moisture': 0.35,
                'Temperature_C': 22.5,
                'Rainfall_mm': 780,
                'Crop_Type': 'Wheat',
                'Fertilizer_Usage_kg': 120,
                'Pesticide_Usage_kg': 5.2,
                'Crop_Yield_ton': 4.8,
                'Sustainability_Score': 7.2
            }
        ]
        
        # Sample market data
        market_data = [
            {
                'Market_ID': 10002,
                'Product': 'Wheat',
                'Market_Price_per_ton': 245.50,
                'Demand_Index': 8.2,
                'Supply_Index': 7.5,
                'Competitor_Price_per_ton': 240.75,
                'Economic_Indicator': 0.85,
                'Weather_Impact_Score': 0.2,
                'Seasonal_Factor': 1.1,
                'Consumer_Trend_Index': 0.95
            }
        ]
        
        # Insert sample data into the database
        for farm in farm_data:
            self.db.insert_farm_data(farm)
        
        for market in market_data:
            self.db.insert_market_data(market)
    
    def generate_recommendations(self, farm_id, market_id):
        """Generate integrated recommendations for a specific farm and market."""
        return self.coordinator.generate_integrated_recommendation(farm_id, market_id)