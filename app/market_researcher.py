from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBRegressor
import ollama

class MarketResearcherAgent:
    """Agent that analyzes market trends and identifies profitable opportunities."""
    
    def __init__(self, database):
        self.db = database
        self.price_models = {}  # Models for different products
        self.demand_models = {}  # Models for demand prediction
        self.train_models()
        
    def train_models(self):
        """Train predictive models based on historical market data."""
        data = self.db.get_all_market_data()
        data.dropna(inplace=True)
        data.fillna(0, inplace=True)
        data['Seasonal_Factor'] = data['Seasonal_Factor'].map({'HIGH': 3, 'MEDIUM': 2, 'LOW': 1})
        if len(data) < 10:  # Not enough data for reliable training
            return
            
        # Train product-specific price prediction models
        for product in data['Product'].unique():
            product_data = data[data['Product'] == product]
            if len(product_data) >= 5:  # Min samples for training
                X = product_data[[ 'Supply_Index', 'Competitor_Price_per_ton',
                                 'Economic_Indicator', 'Weather_Impact_score', 'Seasonal_Factor','Consumer_Trend_Index']]
                y_price = product_data['Market_Price_per_ton']
                y_demand = product_data['Demand_Index']

                # Split the data once - now using y_price but this creates consistent X_train/X_test splits
                X_train, X_test, y_price_train, y_price_test, y_demand_train, y_demand_test = train_test_split(
                    X, y_price, y_demand, test_size=0.2, random_state=42)

                # Scale the features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                # Train price model
                price_model = XGBRegressor( random_state=42)
                price_model.fit(X_train_scaled, y_price_train)

                # Train demand model - using the same X_train_scaled but with y_demand_train
                demand_model = XGBRegressor( random_state=42)
                demand_model.fit(X_train_scaled, y_demand_train)

                # Store both models
                self.price_models[product] = {
                    'model': price_model,
                    'scaler': scaler
                }

                self.demand_models[product] = {
                    'model': demand_model,
                    'scaler': scaler
                }
    
    def predict_price(self, market_data, product):
        """Predict market price based on market conditions."""
        if product not in self.price_models:
            return None  # No model available for this product
        model_info = self.price_models[product]
        features = np.array([[
            # market_data['Demand_Index'], 
            market_data['Supply_Index'], 
            market_data['Competitor_Price_per_ton'],
            market_data['Economic_Indicator'], 
            market_data['Weather_Impact_score'], 
            market_data['Seasonal_Factor'],
            market_data['Consumer_Trend_Index']
        ]])
        
        features_scaled = model_info['scaler'].transform(features)
        predicted_price = model_info['model'].predict(features_scaled)[0]
        
        return predicted_price
    
    def predict_demand(self, market_data, product):
        """Predict market demand based on market conditions."""
        if product not in self.demand_models:
            return None  # No model available for this product
            
        model_info = self.demand_models[product]
        features = np.array([[
            # market_data['Demand_Index'], 
            market_data['Supply_Index'], 
            market_data['Competitor_Price_per_ton'],
            market_data['Economic_Indicator'], 
            market_data['Weather_Impact_score'], 
            market_data['Seasonal_Factor'],
            market_data['Consumer_Trend_Index']
        ]])
        
        features_scaled = model_info['scaler'].transform(features)
        predicted_demand = model_info['model'].predict(features_scaled)[0]
        
        return predicted_demand
    
    def analyze_market_opportunities(self, market_id):
        """Identify market opportunities across different products."""
        market_data = self.db.get_all_market_data()
        market_data.dropna(inplace=True)
        market_data['Seasonal_Factor'] = market_data['Seasonal_Factor'].map({'HIGH': 3, 'MEDIUM': 2, 'LOW': 1})
        if market_data.empty:
            return "Insufficient market data available."
            
        # Get the most recent data for each product provided timestamp is avaialble
        latest_data = market_data.groupby('Product').last().reset_index()
        
        opportunities = []
        for _, row in latest_data.iterrows():
            product = row['Product']
            
            # Skip if no models available
            if product not in self.price_models or product not in self.demand_models:
                continue
                
            # Create market data dict for predictions
            market_dict = {
                # 'Demand_Index': row['Demand_Index'],
                'Supply_Index': row['Supply_Index'],
                'Competitor_Price_per_ton': row['Competitor_Price_per_ton'],
                'Economic_Indicator': row['Economic_Indicator'],
                'Weather_Impact_score': row['Weather_Impact_score'],
                'Seasonal_Factor': row['Seasonal_Factor'],
                'Consumer_Trend_Index': row['Consumer_Trend_Index']
            }
            
            # Make predictions
            price_now = row['Market_Price_per_ton']
            demand_now = row['Demand_Index']
            
            # Predict future (assuming some time horizon)
            # This is a simple approach - in reality, you'd want to forecast with time series
            predicted_price = self.predict_price(market_dict, product)
            predicted_demand = self.predict_demand(market_dict, product)
            
            # Calculate opportunity score
            price_change = (predicted_price - price_now) / price_now
            demand_change = (predicted_demand - demand_now) / demand_now
            opportunity_score = price_change + demand_change  # Simple scoring
            
            opportunities.append({
                'product': product,
                'current_price': price_now,
                'predicted_price': predicted_price,
                'price_change_pct': price_change * 100,
                'current_demand': demand_now,
                'predicted_demand': predicted_demand,
                'demand_change_pct': demand_change * 100,
                'opportunity_score': opportunity_score
            })
        
        # Sort by opportunity score
        opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        return opportunities
    
    def generate_recommendations(self, market_id):
        """Generate market recommendations based on analyzed opportunities."""
        opportunities = self.analyze_market_opportunities(market_id)
        if isinstance(opportunities, str):  # Error message
            return opportunities
        
        if not opportunities:
            return "No market opportunities identified with current data."
        
        # Get top opportunities
        top_opportunities = opportunities[:3]
        
        # Generate text recommendation using LLM
        opportunities_text = "\n".join([
            f"- {op['product']}: Price forecast change: {op['price_change_pct']:.1f}%, " +
            f"Demand forecast change: {op['demand_change_pct']:.1f}%"
            for op in top_opportunities
        ])
        
        prompt = f"""
        Generate a market analysis recommendation for farmers based on these market forecasts:
        
        {opportunities_text}
        
        Include practical advice on:
        1. Which crops show the most promise for profitability
        2. Timing considerations for planting and harvesting
        3. Market positioning and potential premium niches
        
        Keep the recommendation concise and actionable.
        """
        
        try:
            response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            recommendation_text = response['message']['content']
        except Exception as e:
            # Fallback if API call fails
            recommendation_text = f"""
            Based on our market analysis, we recommend focusing on these top opportunities:
            {opportunities_text}
            
            These crops show stronger pricing and demand trends in the coming season.
            """
        
        recommendation = {
            'Farm_ID': 'ALL',  # Market recommendations may apply to all farms
            'recommendation_type': 'market',
            'recommendation_text': recommendation_text,
            'projected_sustainability_impact': 0.0,  # Market analysis doesn't directly address sustainability
            'projected_profit_impact': top_opportunities[0]['opportunity_score'] if top_opportunities else 0.0,
            'confidence_score': 0.7  # Placeholder confidence score
        }
        
        # Store recommendation in database
        self.db.insert_recommendation(recommendation)
        
        return recommendation