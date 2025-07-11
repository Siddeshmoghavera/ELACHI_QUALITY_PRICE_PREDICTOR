import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_elaichi_dataset(n_samples=1000):
    """
    Generate synthetic Elaichi dataset with logical relationships
    between features and target variables.
    """
    data = []
    
    for i in range(n_samples):
        # Generate base features with realistic ranges
        moisture = round(np.random.normal(8.5, 1.5), 2)  # 6-12% typical range
        moisture = max(5.0, min(15.0, moisture))  # Clamp to realistic range
        
        size = round(np.random.normal(12.0, 2.0), 2)  # 8-16mm typical range
        size = max(8.0, min(18.0, size))
        
        color = random.randint(1, 10)
        aroma = random.randint(1, 10)
        
        oil_content = round(np.random.normal(4.5, 1.0), 2)  # 2-7% typical range
        oil_content = max(2.0, min(8.0, oil_content))
        
        # Calculate quality score based on features (weighted combination)
        quality_score = (
            (10 - moisture) * 0.15 +  # Lower moisture is better
            size * 0.25 +            # Larger size is better
            color * 0.25 +           # Higher color score is better
            aroma * 0.25 +           # Higher aroma is better
            oil_content * 0.10       # Higher oil content is better
        )
        
        # Normalize quality score to 0-10 range
        quality_score = max(0, min(10, quality_score))
        
        # Determine quality label based on score
        if quality_score >= 7.5:
            quality_label = "Premium"
            base_price = 2500
        elif quality_score >= 5.5:
            quality_label = "Standard"
            base_price = 1500
        else:
            quality_label = "Low"
            base_price = 800
        
        # Calculate price with some variation and market factors
        price_variation = np.random.normal(0, 200)  # Market fluctuation
        
        # Premium features boost price
        premium_bonus = 0
        if size >= 14:
            premium_bonus += 300
        if color >= 8:
            premium_bonus += 200
        if aroma >= 8:
            premium_bonus += 200
        if oil_content >= 6:
            premium_bonus += 250
        if moisture <= 7:
            premium_bonus += 150
        
        price_per_kg = int(base_price + premium_bonus + price_variation)
        price_per_kg = max(500, min(4000, price_per_kg))  # Realistic price range
        
        data.append({
            'Moisture': moisture,
            'Size': size,
            'Color': color,
            'Aroma': aroma,
            'Oil_Content': oil_content,
            'Price_per_kg': price_per_kg,
            'Quality_Label': quality_label
        })
    
    return pd.DataFrame(data)

# Generate the dataset
print("Generating Elaichi dataset...")
df = generate_elaichi_dataset(1000)

# Display basic statistics
print(f"\nDataset generated with {len(df)} samples")
print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

print("\nQuality Label Distribution:")
print(df['Quality_Label'].value_counts())

print("\nPrice Statistics by Quality:")
print(df.groupby('Quality_Label')['Price_per_kg'].agg(['mean', 'min', 'max']))

# Save the dataset
df.to_csv('elaichi_dataset.csv', index=False)
print("\nDataset saved as 'elaichi_dataset.csv'")

# Display first few rows
print("\nFirst 10 rows of the dataset:")
print(df.head(10))