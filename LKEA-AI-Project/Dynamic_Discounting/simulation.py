import pandas as pd
import numpy as np

# %% Settings
pd.set_option('display.max_columns', None)

# %% Load Data
def load_data(path):
    df = pd.read_excel(path)
    return df

# %% Initial Cleaning
def initial_cleaning(df):

    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values(by='Date')

    expected_product_columns = [
        'AIS(Air Insulated Switchgear)', 'RMU(Ring Main Unit)', 
        'PSS(Compact Sub-Stations)', 'VCU(Vacuum Contactor Units)', 
        'E-House', 'VCB(Vacuum Circuit Breaker)', 
        'ACB(Air Circuit Breaker)', 'MCCB(Moduled Case Circuit Breaker)', 
        'SDF(Switch Disconnectors)', 'BBT(Busbar Trunking)', 
        'Modular Switches', 'Starter', 'Controller', 
        'Solar Solutions', 'Pump Starter and Controller'
    ]

    product_columns = [col for col in expected_product_columns if col in df.columns]
    df['Product_id'] = df[product_columns].idxmax(axis=1)

    df = df[['Partner_id', 'Product_id', 'Date', 'MRP', 'Sales_Quantity_Last_Period', 
             'Discount_Applied', 'Geography', 'Competitor_Price']].dropna()

    df.rename(columns={
        'MRP': 'Price', 
        'Sales_Quantity_Last_Period': 'Demand', 
        'Discount_Applied': 'Discount'
    }, inplace=True)

    df = df[(df['Price'] > 0) & (df['Demand'] > 0)]
    return df

# %% Prepare Monthly Data
def prepare_data(df):
    df['YearMonth'] = pd.to_datetime(df['Date'].dt.to_period('M').astype(str))
    df.set_index('YearMonth', inplace=True)

    monthly_df = df.groupby(['Product_id', 'Geography', 'YearMonth']).agg({
        'Demand': 'sum',
        'Price': 'mean',
        'Competitor_Price': 'mean',
        'Discount': 'mean'
    }).reset_index()

    for col in ['Demand', 'Price', 'Competitor_Price']:
        monthly_df[f'Log_{col}'] = np.log1p(monthly_df[col])

    return monthly_df.dropna()


def calculate_new_demand(demand, elasticity, discount):
    """Calculate demand response to price changes."""
    discount = np.clip(discount, 0, 50)
    if elasticity is None or np.isnan(elasticity):
        elasticity = -0.2
    return demand * np.exp(elasticity * np.log(1 - discount / 100))

def create_price_discount_simulation(product_region_data, elasticity_value, product_name, region_name):
    """
    Create simulation table with:
    - 5 equidistant price buckets between min/max observed prices
    - Average demand calculated for each price bucket
    - Demand projections at 0-50% discounts (5% intervals)
    """
    data = product_region_data[
        (product_region_data['Product_id'] == product_name) &
        (product_region_data['Geography'] == region_name)
    ].copy()

    if data.empty:
        print(f"No data for {product_name} in {region_name}")
        return None

    # Create 4 price buckets (5 edges)
    min_price = data['Price'].min()
    max_price = data['Price'].max()
    price_buckets = np.linspace(min_price, max_price, 5)

    # Assign buckets and calculate stats
    data['price_bucket'] = pd.cut(data['Price'], bins=price_buckets, include_lowest=True)
    bucket_stats = data.groupby('price_bucket').agg(
        avg_demand=('Demand', 'mean'),
        price_midpoint=('Price', lambda x: (x.min() + x.max()) / 2)
    ).reset_index()

    # Discount scenarios (0% to 50% in 5% steps)
    discount_rates = np.arange(0, 55, 5)

    simulation_results = []
    for _, bucket in bucket_stats.iterrows():
        for discount in discount_rates:
            new_demand = calculate_new_demand(
                demand=bucket['avg_demand'],
                elasticity=elasticity_value,
                discount=discount
            )

            simulation_results.append({
                'product': product_name,
                'region': region_name,
                'price_bucket': f"{bucket['price_bucket'].left:.2f}-{bucket['price_bucket'].right:.2f}",
                'price_midpoint': round(bucket['price_midpoint'], 2),
                'discount_pct': discount,
                'original_avg_demand': round(bucket['avg_demand'], 2),
                'predicted_demand': round(new_demand, 2),
                'demand_change_pct': round((new_demand - bucket['avg_demand']) / bucket['avg_demand'] * 100, 1),
                'elasticity_used': round(elasticity_value, 3)
            })

    return pd.DataFrame(simulation_results)

def calculate_optimal_discounts(simulation_df):
    """Calculate profit-optimizing discount at each price point."""
    simulation_df['revenue'] = (
        simulation_df['price_midpoint'] *
        (1 - simulation_df['discount_pct'] / 100) *
        simulation_df['predicted_demand']
    )

    optimal_discounts = (
        simulation_df.loc[simulation_df.groupby('price_bucket')['revenue'].idxmax()]
        [['product', 'region', 'price_bucket', 'price_midpoint',
          'discount_pct', 'predicted_demand', 'revenue']]
        .sort_values('price_midpoint')
        .rename(columns={
            'discount_pct': 'optimal_discount_pct',
            'predicted_demand': 'optimal_demand',
            'revenue': 'max_revenue'
        })
    )

    return optimal_discounts

def save_optimal_discounts(simulation_df, filename="optimal_discounts.csv"):
    """Calculate and save optimal discount scenarios."""
    optimal_df = calculate_optimal_discounts(simulation_df)
    optimal_df['price_midpoint'] = optimal_df['price_midpoint'].round(2)
    optimal_df['max_revenue'] = optimal_df['max_revenue'].round(2)

    optimal_df.to_csv(filename, index=False)
    print(f"Saved optimal discounts to {filename}")
    return optimal_df

if __name__ == "__main__":
    # Load elasticity values
    elasticity_df = pd.read_csv("output_folder/my_product_forecasts001_elasticity.csv")
    product_elasticity = elasticity_df[
        (elasticity_df['product'] == "RMU(Ring Main Unit)") &
        (elasticity_df['region'] == "West")
    ]['price_elasticity'].values[0]

    # Load transaction data
    df = load_data("input_data/stockist_data_with_date.xlsx")
    df = initial_cleaning(df)
    monthly_df = prepare_data(df)

    # Run simulation
    simulation = create_price_discount_simulation(
        product_region_data=monthly_df,
        elasticity_value=product_elasticity,
        product_name="RMU(Ring Main Unit)",
        region_name="West"
    )

    if simulation is not None:
        # Save full simulation
        simulation.to_csv("simulation_output/rmu_west_simulation.csv", index=False)

        # Save optimal discounts
        optimal_discounts = save_optimal_discounts(simulation, filename="simulation_output/rmu_west_optimal_discounts.csv")

        # Display a sample
        print(optimal_discounts.head())
