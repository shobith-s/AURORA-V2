"""
Expand datasets from 18 to 40 for better neural oracle training
Simple addition - no over-engineering
"""
import pandas as pd
import numpy as np


def create_additional_datasets():
    """Add 22 more datasets (18 existing + 22 new = 40 total)"""
    datasets = {}
    
    # 19. Airline passengers (time series)
    datasets['airline_passengers'] = pd.DataFrame({
        'month': pd.date_range('2020-01-01', periods=500, freq='M'),
        'passengers': np.random.poisson(1000, 500) + np.arange(500) * 2,
        'revenue': np.random.lognormal(12, 0.5, 500),
        'satisfaction': np.random.uniform(3, 5, 500),
    })
    
    # 20. Restaurant reviews
    datasets['restaurant_reviews'] = pd.DataFrame({
        'review_id': range(800),
        'rating': np.random.randint(1, 6, 800),
        'price_range': np.random.choice(['$', '$$', '$$$', '$$$$'], 800),
        'cuisine': np.random.choice(['Italian', 'Chinese', 'Mexican', 'Indian'], 800),
        'delivery_time': np.random.normal(30, 10, 800).clip(10, 90),
    })
    
    # 21. Fitness tracker data
    datasets['fitness_tracker'] = pd.DataFrame({
        'user_id': range(600),
        'steps': np.random.poisson(8000, 600),
        'calories': np.random.normal(2000, 500, 600).clip(1200, 4000),
        'heart_rate': np.random.normal(75, 15, 600).clip(50, 150),
        'sleep_hours': np.random.normal(7, 1.5, 600).clip(4, 12),
    })
    
    # 22. Movie ratings
    datasets['movie_ratings'] = pd.DataFrame({
        'movie_id': range(1000),
        'rating': np.random.uniform(1, 10, 1000),
        'votes': np.random.lognormal(8, 2, 1000).astype(int),
        'budget': np.random.lognormal(17, 1.5, 1000),
        'revenue': np.random.lognormal(18, 2, 1000),
        'genre': np.random.choice(['Action', 'Drama', 'Comedy', 'Thriller'], 1000),
    })
    
    # 23. Hotel bookings
    datasets['hotel_bookings'] = pd.DataFrame({
        'booking_id': range(900),
        'lead_time': np.random.poisson(30, 900),
        'nights': np.random.poisson(3, 900).clip(1, 14),
        'adults': np.random.poisson(2, 900).clip(1, 4),
        'children': np.random.poisson(0.5, 900).clip(0, 3),
        'price_per_night': np.random.lognormal(5, 0.5, 900),
        'canceled': np.random.choice([0, 1], 900, p=[0.7, 0.3]),
    })
    
    # 24. Spotify songs
    datasets['spotify_songs'] = pd.DataFrame({
        'song_id': range(1200),
        'duration_ms': np.random.normal(210000, 60000, 1200).clip(60000, 600000).astype(int),
        'danceability': np.random.uniform(0, 1, 1200),
        'energy': np.random.uniform(0, 1, 1200),
        'loudness': np.random.normal(-6, 3, 1200),
        'tempo': np.random.normal(120, 30, 1200).clip(60, 200),
        'popularity': np.random.beta(2, 5, 1200) * 100,
    })
    
    # 25. Bike sharing
    datasets['bike_sharing'] = pd.DataFrame({
        'hour': np.tile(range(24), 50),
        'temp': np.random.normal(20, 10, 1200),
        'humidity': np.random.uniform(20, 90, 1200),
        'windspeed': np.random.gamma(2, 5, 1200),
        'casual_users': np.random.poisson(50, 1200),
        'registered_users': np.random.poisson(200, 1200),
    })
    
    # 26. Wine quality
    datasets['wine_quality'] = pd.DataFrame({
        'wine_id': range(700),
        'fixed_acidity': np.random.normal(7, 1.5, 700).clip(4, 15),
        'volatile_acidity': np.random.normal(0.5, 0.2, 700).clip(0, 2),
        'citric_acid': np.random.normal(0.3, 0.2, 700).clip(0, 1),
        'residual_sugar': np.random.gamma(2, 3, 700),
        'pH': np.random.normal(3.3, 0.2, 700).clip(2.5, 4),
        'quality': np.random.randint(3, 9, 700),
    })
    
    # 27. Supermarket sales
    datasets['supermarket_sales'] = pd.DataFrame({
        'invoice_id': range(1000),
        'branch': np.random.choice(['A', 'B', 'C'], 1000),
        'product_line': np.random.choice(['Food', 'Electronics', 'Fashion', 'Home'], 1000),
        'unit_price': np.random.lognormal(3, 1, 1000),
        'quantity': np.random.poisson(5, 1000).clip(1, 20),
        'tax': np.random.uniform(0, 50, 1000),
        'rating': np.random.uniform(4, 10, 1000),
    })
    
    # 28. Diabetes prediction
    datasets['diabetes_prediction'] = pd.DataFrame({
        'patient_id': range(500),
        'pregnancies': np.random.poisson(3, 500).clip(0, 15),
        'glucose': np.random.normal(120, 30, 500).clip(50, 200),
        'blood_pressure': np.random.normal(70, 20, 500).clip(40, 120),
        'skin_thickness': np.random.normal(20, 15, 500).clip(0, 100),
        'insulin': np.random.gamma(2, 50, 500),
        'bmi': np.random.normal(32, 7, 500).clip(15, 60),
        'outcome': np.random.choice([0, 1], 500, p=[0.65, 0.35]),
    })
    
    # 29. Mushroom classification
    datasets['mushroom_classification'] = pd.DataFrame({
        'mushroom_id': range(800),
        'cap_diameter': np.random.gamma(2, 3, 800),
        'cap_shape': np.random.choice(['bell', 'conical', 'convex', 'flat'], 800),
        'cap_color': np.random.choice(['brown', 'red', 'white', 'yellow'], 800),
        'stem_height': np.random.normal(10, 3, 800).clip(3, 20),
        'stem_width': np.random.normal(2, 0.5, 800).clip(0.5, 5),
        'edible': np.random.choice([0, 1], 800, p=[0.5, 0.5]),
    })
    
    # 30. Heart disease
    datasets['heart_disease'] = pd.DataFrame({
        'patient_id': range(600),
        'age': np.random.normal(55, 10, 600).clip(30, 80).astype(int),
        'sex': np.random.choice([0, 1], 600),
        'chest_pain_type': np.random.randint(0, 4, 600),
        'resting_bp': np.random.normal(130, 20, 600).clip(90, 200),
        'cholesterol': np.random.normal(240, 50, 600).clip(100, 400),
        'max_heart_rate': np.random.normal(150, 20, 600).clip(80, 200),
        'disease': np.random.choice([0, 1], 600, p=[0.55, 0.45]),
    })
    
    # 31. Spam detection
    datasets['spam_detection'] = pd.DataFrame({
        'email_id': range(1000),
        'word_freq_make': np.random.gamma(1, 0.5, 1000),
        'word_freq_address': np.random.gamma(1, 0.3, 1000),
        'word_freq_all': np.random.gamma(1, 0.4, 1000),
        'word_freq_free': np.random.gamma(1, 0.6, 1000),
        'char_freq_dollar': np.random.gamma(1, 0.2, 1000),
        'capital_run_length_avg': np.random.gamma(2, 10, 1000),
        'is_spam': np.random.choice([0, 1], 1000, p=[0.6, 0.4]),
    })
    
    # 32. Penguin species
    datasets['penguin_species'] = pd.DataFrame({
        'penguin_id': range(400),
        'bill_length_mm': np.random.normal(44, 5, 400).clip(30, 60),
        'bill_depth_mm': np.random.normal(17, 2, 400).clip(13, 22),
        'flipper_length_mm': np.random.normal(200, 15, 400).clip(170, 230),
        'body_mass_g': np.random.normal(4200, 800, 400).clip(2500, 6500),
        'sex': np.random.choice(['male', 'female'], 400),
        'species': np.random.choice(['Adelie', 'Chinstrap', 'Gentoo'], 400),
    })
    
    # 33. Water quality
    datasets['water_quality'] = pd.DataFrame({
        'sample_id': range(500),
        'ph': np.random.normal(7, 1, 500).clip(0, 14),
        'hardness': np.random.normal(200, 50, 500).clip(50, 400),
        'solids': np.random.normal(20000, 5000, 500).clip(5000, 50000),
        'chloramines': np.random.normal(7, 2, 500).clip(0, 15),
        'sulfate': np.random.normal(330, 50, 500).clip(100, 500),
        'conductivity': np.random.normal(400, 100, 500).clip(200, 800),
        'potability': np.random.choice([0, 1], 500, p=[0.6, 0.4]),
    })
    
    # 34. Air quality
    datasets['air_quality'] = pd.DataFrame({
        'station_id': range(600),
        'pm25': np.random.gamma(2, 20, 600),
        'pm10': np.random.gamma(2, 30, 600),
        'no2': np.random.gamma(2, 15, 600),
        'so2': np.random.gamma(2, 10, 600),
        'co': np.random.gamma(2, 0.5, 600),
        'ozone': np.random.gamma(2, 25, 600),
        'aqi': np.random.normal(100, 50, 600).clip(0, 500),
    })
    
    # 35. Crop recommendation
    datasets['crop_recommendation'] = pd.DataFrame({
        'field_id': range(400),
        'nitrogen': np.random.normal(50, 20, 400).clip(0, 140),
        'phosphorus': np.random.normal(50, 20, 400).clip(5, 145),
        'potassium': np.random.normal(50, 20, 400).clip(5, 205),
        'temperature': np.random.normal(25, 5, 400).clip(8, 44),
        'humidity': np.random.uniform(14, 100, 400),
        'ph': np.random.normal(6.5, 0.5, 400).clip(3.5, 9.9),
        'rainfall': np.random.gamma(3, 50, 400),
    })
    
    # 36. Loan approval
    datasets['loan_approval'] = pd.DataFrame({
        'applicant_id': range(700),
        'gender': np.random.choice(['Male', 'Female'], 700),
        'married': np.random.choice(['Yes', 'No'], 700, p=[0.65, 0.35]),
        'dependents': np.random.poisson(1, 700).clip(0, 3),
        'education': np.random.choice(['Graduate', 'Not Graduate'], 700, p=[0.8, 0.2]),
        'self_employed': np.random.choice(['Yes', 'No'], 700, p=[0.15, 0.85]),
        'applicant_income': np.random.lognormal(9, 0.8, 700),
        'loan_amount': np.random.lognormal(11, 0.5, 700),
        'approved': np.random.choice([0, 1], 700, p=[0.3, 0.7]),
    })
    
    # 37. Stroke prediction
    datasets['stroke_prediction'] = pd.DataFrame({
        'patient_id': range(500),
        'age': np.random.normal(50, 15, 500).clip(18, 90).astype(int),
        'hypertension': np.random.choice([0, 1], 500, p=[0.9, 0.1]),
        'heart_disease': np.random.choice([0, 1], 500, p=[0.95, 0.05]),
        'avg_glucose_level': np.random.normal(110, 40, 500).clip(50, 300),
        'bmi': np.random.normal(28, 7, 500).clip(15, 60),
        'smoking_status': np.random.choice(['never', 'formerly', 'smokes'], 500),
        'stroke': np.random.choice([0, 1], 500, p=[0.95, 0.05]),
    })
    
    # 38. Car price prediction
    datasets['car_price'] = pd.DataFrame({
        'car_id': range(800),
        'year': np.random.randint(2000, 2024, 800),
        'mileage': np.random.lognormal(10, 1, 800),
        'engine_size': np.random.uniform(1.0, 5.0, 800),
        'horsepower': np.random.normal(200, 50, 800).clip(80, 500),
        'fuel_type': np.random.choice(['Petrol', 'Diesel', 'Electric'], 800),
        'transmission': np.random.choice(['Manual', 'Automatic'], 800, p=[0.3, 0.7]),
        'price': np.random.lognormal(10, 0.8, 800),
    })
    
    # 39. Online shoppers intention
    datasets['online_shoppers'] = pd.DataFrame({
        'session_id': range(1000),
        'administrative_duration': np.random.gamma(2, 50, 1000),
        'informational_duration': np.random.gamma(2, 30, 1000),
        'product_related_duration': np.random.gamma(3, 200, 1000),
        'bounce_rates': np.random.uniform(0, 0.2, 1000),
        'exit_rates': np.random.uniform(0, 0.2, 1000),
        'page_values': np.random.gamma(2, 10, 1000),
        'revenue': np.random.choice([0, 1], 1000, p=[0.85, 0.15]),
    })
    
    # 40. Forest fire prediction
    datasets['forest_fire'] = pd.DataFrame({
        'fire_id': range(500),
        'temperature': np.random.normal(20, 8, 500).clip(0, 40),
        'relative_humidity': np.random.uniform(15, 100, 500),
        'wind_speed': np.random.gamma(2, 3, 500),
        'rain': np.random.gamma(1, 2, 500),
        'ffmc': np.random.uniform(18, 97, 500),  # Fine Fuel Moisture Code
        'dmc': np.random.uniform(1, 300, 500),   # Duff Moisture Code
        'dc': np.random.uniform(7, 900, 500),    # Drought Code
        'isi': np.random.uniform(0, 20, 500),    # Initial Spread Index
        'area': np.random.gamma(1, 10, 500),     # Burned area
    })
    
    return datasets


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add to existing datasets
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from validator.scripts.download_datasets import create_diverse_synthetic_datasets
    
    print("Creating 40 datasets total...")
    print("  - 18 existing datasets")
    print("  - 22 new datasets")
    
    # Get existing 18
    existing = create_diverse_synthetic_datasets()
    print(f"\n✅ Loaded {len(existing)} existing datasets")
    
    # Add new 22
    new_datasets = create_additional_datasets()
    print(f"✅ Created {len(new_datasets)} new datasets")
    
    # Combine
    all_datasets = {**existing, **new_datasets}
    print(f"\n✅ TOTAL: {len(all_datasets)} datasets")
    
    # Save
    output_dir = Path('validator/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, df in all_datasets.items():
        output_path = output_dir / f"{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"  Saved {name}: {df.shape}")
    
    print(f"\n✅ All 40 datasets saved to {output_dir}")
