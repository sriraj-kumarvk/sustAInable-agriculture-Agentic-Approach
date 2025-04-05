import sqlite3
from datetime import datetime
import pandas as pd

class Database:
    """SQLite database for storing and retrieving farming and market data."""
    
    def __init__(self, db_path="farming_system.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.initialize_database()
        
    def initialize_database(self):
        """Create necessary tables if they don't exist."""
        # Farmer advisor data table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS farm_data (
            Farm_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Soil_pH REAL,
            Soil_Moisture REAL,
            Temperature_C REAL,
            Rainfall_mm REAL,
            Crop_Type TEXT,
            Fertilizer_Usage_kg REAL,
            Pesticide_Usage_kg REAL,
            Crop_Yield_ton REAL,
            Sustainability_Score REAL
        )
        ''')
        
        # Market research data table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            Market_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Product TEXT,
            Market_Price_per_ton REAL,
            Demand_Index REAL,
            Supply_Index REAL,
            Competitor_Price_per_ton REAL,
            Economic_Indicator REAL,
            Weather_Impact_Score REAL,
            Seasonal_Factor REAL,
            Consumer_Trend_Index REAL
        )
        ''')
        
        # Recommendations table for storing agent outputs
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            farm_id TEXT,
            recommendation_type TEXT,
            recommendation_text TEXT,
            projected_sustainability_impact REAL,
            projected_profit_impact REAL,
            confidence_score REAL
        )
        ''')
        
        self.conn.commit()

    def import_from_csv(self, csv_path, table_type):
        try:
            # Read CSV file
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            if table_type == 'farm':
                # Check required columns for farm data
                required_cols = ['Farm_ID', 'Soil_pH', 'Soil_Moisture', 'Temperature_C', 
                                'Rainfall_mm', 'Crop_Type', 'Fertilizer_Usage_kg', 
                                'Pesticide_Usage_kg', 'Crop_Yield_ton', 'Sustainability_Score']
                
                # Verify all required columns are present
                if not all(col in df.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in df.columns]
                    return f"Error: Missing required columns: {missing}"
                
                # Insert records
                for _, row in df.iterrows():
                    
                    self.insert_farm_data({
                        'Farm_ID': row['Farm_ID'],
                        'Soil_pH': row['Soil_pH'],
                        'Soil_Moisture': row['Soil_Moisture'],
                        'Temperature_C': row['Temperature_C'],
                        'Rainfall_mm': row['Rainfall_mm'],
                        'Crop_Type': row['Crop_Type'],
                        'Fertilizer_Usage_kg': row['Fertilizer_Usage_kg'],
                        'Pesticide_Usage_kg': row['Pesticide_Usage_kg'],
                        'Crop_Yield_ton': row['Crop_Yield_ton'],
                        'Sustainability_Score': row['Sustainability_Score']
                    })
                
                    
            elif table_type == 'market':
                # Check required columns for market data
                required_cols = ['Market_ID', 'Product', 'Market_Price_per_ton', 
                                'Demand_Index', 'Supply_Index', 'Competitor_Price_per_ton',
                                'Economic_Indicator', 'Weather_Impact_Score', 
                                'Seasonal_Factor', 'Consumer_Trend_Index']
                
                # Verify all required columns are present
                if not all(col in df.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in df.columns]
                    return f"Error: Missing required columns: {missing}"
                    
                # Insert records
                for _, row in df.iterrows():
                    self.insert_market_data({
                        'Market_ID': row['Market_ID'],
                        'Product': row['Product'],
                        'Market_Price_per_ton': row['Market_Price_per_ton'],
                        'Demand_Index': row['Demand_Index'],
                        'Supply_Index': row['Supply_Index'],
                        'Competitor_Price_per_ton': row['Competitor_Price_per_ton'],
                        'Economic_Indicator': row['Economic_Indicator'],
                        'Weather_Impact_Score': row['Weather_Impact_Score'],
                        'Seasonal_Factor': row['Seasonal_Factor'],
                        'Consumer_Trend_Index': row['Consumer_Trend_Index']
                    })
            else:
                return "Error: table_type must be either 'farm' or 'market'"
                
            return f"Successfully imported {len(df)} records into {table_type} table"
            
        except Exception as e:
            return f"Error importing data: {str(e)}"
        
    def insert_farm_data(self, data):
        """Insert new farm data into the database."""
        now = datetime.now().isoformat()
        query = '''
        INSERT INTO farm_data (
             Farm_ID, Soil_pH, Soil_Moisture, Temperature_C, 
            Rainfall_mm, Crop_Type, Fertilizer_Usage_kg, Pesticide_Usage_kg, 
            Crop_Yield_ton, Sustainability_Score
        ) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        self.cursor.execute(query, (
            data['Farm_ID'], data['Soil_pH'], data['Soil_Moisture'], 
            data['Temperature_C'], data['Rainfall_mm'], data['Crop_Type'], 
            data['Fertilizer_Usage_kg'], data['Pesticide_Usage_kg'], 
            data['Crop_Yield_ton'], data['Sustainability_Score']
        ))
        self.conn.commit()
        
    def insert_market_data(self, data):
        """Insert new market data into the database."""
        now = datetime.now().isoformat()
        query = '''
        INSERT INTO market_data (
            Market_ID, Product, Market_Price_per_ton, Demand_Index, 
            Supply_Index, Competitor_Price_per_ton, Economic_Indicator,
            Weather_Impact_Score, Seasonal_Factor, Consumer_Trend_Index
        ) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        self.cursor.execute(query, (
            data['Market_ID'], data['Product'], data['Market_Price_per_ton'], 
            data['Demand_Index'], data['Supply_Index'], data['Competitor_Price_per_ton'], 
            data['Economic_Indicator'], data['Weather_Impact_Score'], 
            data['Seasonal_Factor'], data['Consumer_Trend_Index']
        ))
        self.conn.commit()
        
    def insert_recommendation(self, data):
        """Store a new recommendation in the database."""
        now = datetime.now().isoformat()
        query = '''
        INSERT INTO recommendations (
            timestamp, Farm_ID, recommendation_type, recommendation_text,
            projected_sustainability_impact, projected_profit_impact, confidence_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        self.cursor.execute(query, (
            now, data['Farm_ID'], data['recommendation_type'], data['recommendation_text'],
            data['projected_sustainability_impact'], data['projected_profit_impact'], 
            data['confidence_score']
        ))
        self.conn.commit()
        
    def get_farm_history(self, farm_id, limit=10):
        """Retrieve historical farm data for a specific farm."""
        query = '''
        SELECT * FROM farm_data 
        WHERE farm_id = ? 
        LIMIT ?
        '''
        self.cursor.execute(query, (farm_id, limit))
        columns = [col[0] for col in self.cursor.description]
        results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        return results
    
    def get_market_trends(self, product, limit=10):
        """Retrieve historical market data for a specific product."""
        query = '''
        SELECT * FROM market_data 
        WHERE product = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        '''
        self.cursor.execute(query, (product, limit))
        columns = [col[0] for col in self.cursor.description]
        results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        return results
    
    def get_all_farm_data(self):
        """Retrieve all farm data for model training."""
        query = "SELECT * FROM farm_data"
        return pd.read_sql_query(query, self.conn)
    
    def get_all_market_data(self):
        """Retrieve all market data for model training."""
        query = "SELECT * FROM market_data"
        return pd.read_sql_query(query, self.conn)

    def close(self):
        """Close the database connection."""
        self.conn.close()