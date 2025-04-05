class CoordinatorAgent:
    """Integrates recommendations from specialized agents to provide holistic advice."""
    
    def __init__(self, database, farmer_advisor, market_researcher):
        self.db = database
        self.farmer_advisor = farmer_advisor
        self.market_researcher = market_researcher
    
    def generate_integrated_recommendation(self, farm_id, market_id):
        """Generate integrated recommendations combining farm and market insights."""
        # Get specialized recommendations
        farm_recommendation = self.farmer_advisor.generate_recommendations(farm_id)
        market_recommendation = self.market_researcher.generate_recommendations(market_id)
        
        if isinstance(farm_recommendation, str) or isinstance(market_recommendation, str):
            error_message = []
            if isinstance(farm_recommendation, str):
                error_message.append(f"Farm advisor error: {farm_recommendation}")
            if isinstance(market_recommendation, str):
                error_message.append(f"Market researcher error: {market_recommendation}")
            return "\n".join(error_message)
        
        # Get recent farm data for context
        farm_history = self.db.get_farm_history(farm_id, limit=1)
        if not farm_history:
            return "Cannot generate integrated recommendation: No data for farm " + farm_id
        
        farm_data = farm_history[0]
        
        # Create prompt for integrated recommendation
        prompt = f"""
        Create an integrated farming recommendation that balances sustainability and profitability.
        
        FARM CONTEXT:
        - Farm ID: {farm_id}
        - Current crop: {farm_data['Crop_Type']}
        - Current sustainability score: {farm_data['Sustainability_Score']}
        
        SUSTAINABILITY RECOMMENDATION:
        {farm_recommendation['recommendation_text']}
        
        MARKET RECOMMENDATION:
        {market_recommendation['recommendation_text']}
        
        Create a balanced recommendation that:
        1. Prioritizes long-term sustainability while being economically viable
        2. Identifies specific actions the farmer should take
        3. Explains expected outcomes and benefits
        4. Acknowledges potential challenges and how to address them
        
        The recommendation should be practical, specific, and actionable.
        """
        
        try:
            response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            integrated_text = response['message']['content']
            
        except Exception as e:
            # Fallback if API call fails
            integrated_text = f"""
            INTEGRATED RECOMMENDATION:
            
            SUSTAINABILITY: {farm_recommendation['recommendation_text']}
            
            MARKET CONSIDERATIONS: {market_recommendation['recommendation_text']}
            
            We recommend implementing the sustainability practices above while considering the market trends noted.
            """
        
        # Calculate weighted impact scores
        sustainability_weight = 0.6  # Prioritize sustainability
        profit_weight = 0.4
        
        integrated_sustainability_impact = (
            farm_recommendation['projected_sustainability_impact'] * sustainability_weight + 
            market_recommendation['projected_sustainability_impact'] * profit_weight
        )
        
        integrated_profit_impact = (
            farm_recommendation['projected_profit_impact'] * sustainability_weight + 
            market_recommendation['projected_profit_impact'] * profit_weight
        )
        
        # Average confidence scores
        confidence = (farm_recommendation['confidence_score'] + market_recommendation['confidence_score']) / 2
        
        recommendation = {
            'Farm_ID': farm_id,
            'recommendation_type': 'integrated',
            'recommendation_text': integrated_text,
            'projected_sustainability_impact': integrated_sustainability_impact,
            'projected_profit_impact': integrated_profit_impact,
            'confidence_score': confidence
        }
        
        # Store recommendation in database
        self.db.insert_recommendation(recommendation)
        
        return recommendation
