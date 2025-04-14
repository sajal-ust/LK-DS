# %% Imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

# %% Train-Test Split
def train_test_split_monthly(df, test_size=0.2):
    df['YearMonth'] = pd.to_datetime(df['YearMonth'])
    unique_months = pd.Series(df['YearMonth'].unique()).sort_values().values
    split_idx = int(len(unique_months) * (1 - test_size))
    train_months = unique_months[:split_idx]
    test_months = unique_months[split_idx:]
    train_df = df[df['YearMonth'].isin(train_months)]
    test_df = df[df['YearMonth'].isin(test_months)]
    return train_df, test_df

# %% SARIMAX Training
def train_sarimax_model(train_group):
    try:
        train_group = train_group.copy().sort_values('YearMonth')
        train_group['YearMonth'] = pd.to_datetime(train_group['YearMonth'])
        train_group.set_index('YearMonth', inplace=True)
        train_group = train_group.asfreq('MS')

        y_train = train_group['Log_Demand']
        exog_cols = ['Log_Price', 'Log_Competitor_Price']
        exog_train = train_group[exog_cols].ffill()

        if len(y_train) < 24:
            print("Skipping training: Insufficient data.")
            return None

        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=True,
            enforce_invertibility=True
        )
        fitted_model = model.fit()
        return fitted_model
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return None

# %% Elasticity Calculation
def calculate_elasticity(fitted_model):
    try:
        if hasattr(fitted_model, 'params') and 'Log_Price' in fitted_model.params:
            return fitted_model.params['Log_Price']
        return np.nan
    except Exception as e:
        print(f"Elasticity calculation failed: {str(e)}")
        return np.nan

# %% Evaluation Placeholder
def evaluate_model(model, test_group):
    # Implementation needed
    pass

# Example usage (uncomment to run interactively)
df = load_data("stockist_data_with_date3.xlsx")
df = initial_cleaning(df)
monthly_df = prepare_data(df)
train_df, test_df = train_test_split_monthly(monthly_df)
fitted_model = train_sarimax_model(train_df)
elasticity = calculate_elasticity(fitted_model)
print("Elasticity:", elasticity)
