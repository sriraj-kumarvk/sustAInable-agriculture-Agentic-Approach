from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBRegressor
import ollama

class FarmerAdvisorAgent:
    """Agent that analyzes farm conditions and provides sustainability-focused advice."""
    
    def __init__(self, database):
        self.db = database
        self.crop_models = {}  # Models for different crops
        self.sustainability_model = None
        self.train_models()
        
    def train_models(self):
        """Train predictive models based on historical farm data."""
        data = self.db.get_all_farm_data()
        data.dropna(inplace=True)
        if len(data) < 10:  # Not enough data for reliable training
            return
            
        # Training models for crop-specific yield prediction
        for crop in data['Crop_Type'].unique():
            crop_data = data[data['Crop_Type'] == crop]
            if len(crop_data) >= 5:  # Minimum samples for training
                X = crop_data[['Soil_pH', 'Soil_Moisture', 'Temperature_C', 
                              'Rainfall_mm', 'Fertilizer_Usage_kg', 'Pesticide_Usage_kg']]
                y = crop_data['Crop_Yield_ton']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                model = XGBRegressor( random_state=42)
                model.fit(X_train_scaled, y_train)
                
                self.crop_models[crop] = {
                    'model': model,
                    'scaler': scaler
                }
        
        # Train sustainability score prediction model
        X = data[['Soil_pH', 'Soil_Moisture', 'Temperature_C', 
                              'Rainfall_mm', 'Fertilizer_Usage_kg', 'Pesticide_Usage_kg', 'Crop_Yield_ton']]
        y = data['Sustainability_Score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = XGBRegressor( random_state=42)
        model.fit(X_train_scaled, y_train)
        
        self.sustainability_model = {
            'model': model,
            'scaler': scaler
        }
    
    def predict_yield(self, farm_data, Crop_Type):
        """Predict crop yield based on farm conditions."""
        if Crop_Type not in self.crop_models:
            return None  # No model available for this crop
            
        model_info = self.crop_models[Crop_Type]
        features = np.array([[
            farm_data['Soil_pH'], 
            farm_data['Soil_Moisture'], 
            farm_data['Temperature_C'],
            farm_data['Rainfall_mm'], 
            farm_data['Fertilizer_Usage_kg'], 
            farm_data['Pesticide_Usage_kg']
        ]])
        
        features_scaled = model_info['scaler'].transform(features)
        predicted_yield = model_info['model'].predict(features_scaled)[0]
        
        return predicted_yield
    
    def predict_sustainability(self, farm_data, predicted_yield):
        """Predict sustainability score based on farm practices and yield."""
        if self.sustainability_model is None:
            return None  # Model not available
            
        features = np.array([[
            farm_data['Soil_pH'], 
            farm_data['Soil_Moisture'], 
            farm_data['Temperature_C'],
            farm_data['Rainfall_mm'], 
            farm_data['Fertilizer_Usage_kg'], 
            farm_data['Pesticide_Usage_kg'],
            predicted_yield
        ]])
        
        features_scaled = self.sustainability_model['scaler'].transform(features)
        predicted_score = self.sustainability_model['model'].predict(features_scaled)[0]
        
        return predicted_score
    
    def optimize_inputs(self, farm_data, Crop_Type):
        """Find optimal fertilizer and pesticide usage for sustainability."""
        if Crop_Type not in self.crop_models or self.sustainability_model is None:
            return farm_data['Fertilizer_Usage_kg'], farm_data['Pesticide_Usage_kg']
            
        best_sustainability = 0
        best_fertilizer = farm_data['Fertilizer_Usage_kg']
        best_pesticide = farm_data['Pesticide_Usage_kg']
        current_yield = self.predict_yield(farm_data, Crop_Type)
        
        # Grid search for optimal inputs (simplified approach)
        for fertilizer_factor in [0.7, 0.8, 0.9, 1.0]:
            for pesticide_factor in [0.7, 0.8, 0.9, 1.0]:
                test_data = farm_data.copy()
                test_data['Fertilizer_Usage_kg'] *= fertilizer_factor
                test_data['Pesticide_Usage_kg'] *= pesticide_factor
                
                predicted_yield = self.predict_yield(test_data, Crop_Type)
                
                # Only consider options that maintain at least 90% of current yield
                if predicted_yield >= 0.9 * current_yield:
                    sustainability = self.predict_sustainability(test_data, predicted_yield)
                    if sustainability > best_sustainability:
                        best_sustainability = sustainability
                        best_fertilizer = test_data['Fertilizer_Usage_kg']
                        best_pesticide = test_data['Pesticide_Usage_kg']
        
        return best_fertilizer, best_pesticide
    
    def generate_recommendations(self, farm_id):
        """Generate sustainable farming recommendations for a specific farm."""
        # Get recent farm data
        farm_history = self.db.get_farm_history(farm_id, limit=1)
        if not farm_history:
            return "Insufficient data for farm " + str(farm_id)
            
        farm_data = farm_history[0]
        Crop_Type = farm_data['Crop_Type']
        
        # Generate optimized input recommendations
        optimal_fertilizer, optimal_pesticide = self.optimize_inputs(farm_data, Crop_Type)
        
        # Calculate expected impact
        original_yield = self.predict_yield(farm_data, Crop_Type)
        original_sustainability = farm_data['Sustainability_Score']
        
        optimized_farm_data = farm_data.copy()
        optimized_farm_data['Fertilizer_Usage_kg'] = optimal_fertilizer
        optimized_farm_data['Pesticide_Usage_kg'] = optimal_pesticide
        
        new_yield = self.predict_yield(optimized_farm_data, Crop_Type)
        new_sustainability = self.predict_sustainability(optimized_farm_data, new_yield)
        
        # Generate text recommendation using LLM
        prompt = f"""
        Generate a farming recommendation focused on sustainability for a farmer with the following data:
        - Current crop: {Crop_Type}
        - Current fertilizer usage: {farm_data['Fertilizer_Usage_kg']} kg
        - Recommended fertilizer usage: {optimal_fertilizer} kg
        - Current pesticide usage: {farm_data['Pesticide_Usage_kg']} kg
        - Recommended pesticide usage: {optimal_pesticide} kg
        - Expected yield change: {(new_yield - original_yield) / original_yield * 100:.1f}%
        - Expected sustainability improvement: {(new_sustainability - original_sustainability) / original_sustainability * 100:.1f}%
        
        Focus on practical advice for implementing these changes and explain the environmental benefits.
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
            # in case of API call fails
            recommendation_text = f"""
            Based on our analysis, we recommend:
            - Reduce fertilizer usage from {farm_data['Fertilizer_Usage_kg']} kg to {optimal_fertilizer} kg
            - Reduce pesticide usage from {farm_data['Pesticide_Usage_kg']} kg to {optimal_pesticide} kg
            These changes can maintain yield while improving sustainability.
            """
        
        recommendation = {
            'Farm_ID': farm_id,
            'recommendation_type': 'sustainability',
            'recommendation_text': recommendation_text,
            'projected_sustainability_impact': (new_sustainability - original_sustainability) / original_sustainability,
            'projected_profit_impact': (new_yield - original_yield) / original_yield,
            'confidence_score': 0.8  # Placeholder confidence score
        }
        
        # Store recommendation in database
        self.db.insert_recommendation(recommendation)
        
        return recommendation