"""
Generate sample flight cargo data for testing
"""
import pandas as pd
import numpy as np
import os

def generate_sample_data(n_samples=1000, output_file='sample_flight_data.csv'):
    """Generate synthetic flight cargo data"""
    np.random.seed(42)
    
    # Passenger info
    passenger_ages = np.random.randint(18, 80, n_samples)
    passenger_weights = np.random.normal(70, 15, n_samples)
    passenger_weights = np.clip(passenger_weights, 45, 120)
    
    # Flight info
    flight_durations = np.random.choice([2, 4, 6, 8, 12], n_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05])
    flight_types = np.random.choice(['domestic', 'international', 'regional'], n_samples, p=[0.5, 0.3, 0.2])
    seat_classes = np.random.choice(['economy', 'business', 'first'], n_samples, p=[0.7, 0.25, 0.05])
    
    # Travel info
    number_of_bags = np.random.poisson(1.5, n_samples)
    number_of_bags = np.clip(number_of_bags, 0, 5)
    travel_purpose = np.random.choice(['business', 'leisure', 'family'], n_samples, p=[0.4, 0.4, 0.2])
    
    # Season
    seasons = np.random.choice(['spring', 'summer', 'fall', 'winter'], n_samples)
    
    # Generate luggage size based on relationships
    # Base size
    luggage_size = 20 + passenger_ages * 0.1
    
    # Add effects
    luggage_size += flight_durations * 2  # Longer flights = more luggage
    luggage_size += number_of_bags * 15  # More bags = more total size
    luggage_size += np.where(seat_classes == 'first', 10, 
                    np.where(seat_classes == 'business', 5, 0))  # Higher class = more luggage
    luggage_size += np.where(travel_purpose == 'leisure', 8,
                    np.where(travel_purpose == 'family', 12, 0))  # Leisure/family = more luggage
    luggage_size += np.where(seasons == 'winter', 5, 0)  # Winter = more clothes
    
    # Add some noise
    luggage_size += np.random.normal(0, 5, n_samples)
    luggage_size = np.clip(luggage_size, 10, 100)  # Reasonable range
    
    # Create DataFrame
    df = pd.DataFrame({
        'passenger_age': passenger_ages,
        'passenger_weight': passenger_weights,
        'flight_duration': flight_durations,
        'flight_type': flight_types,
        'seat_class': seat_classes,
        'number_of_bags': number_of_bags,
        'travel_purpose': travel_purpose,
        'season': seasons,
        'luggage_size': luggage_size
    })
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', output_file)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} samples and saved to {output_path}")
    return df

if __name__ == '__main__':
    generate_sample_data(n_samples=1000)



