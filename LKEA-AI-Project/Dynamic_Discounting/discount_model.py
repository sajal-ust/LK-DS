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
    """Evaluate model performance on test data"""
    try:
        # Clean test data
        # test_group = clean_log_values(test_group.copy())
        test_group = test_group.sort_values('YearMonth')
        
        # Prepare test data
        y_test = test_group.set_index('YearMonth')['Log_Demand']
        exog_test = test_group.set_index('YearMonth')[['Log_Price', 'Log_Competitor_Price']]
        
        # Generate forecasts
        forecast = model.get_forecast(
            steps=len(y_test),
            exog=exog_test
        )
        
        # Calculate metrics (converting back from log scale)
        pred = np.exp(forecast.predicted_mean).values  # Convert to numpy array
        actual = np.exp(y_test).values  # Convert to numpy array
        
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", RuntimeWarning)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'actual': actual,
            'predicted': pred
        }
    
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return None
    

def generate_forecasts(model, train_group, steps=12):
    """Generate future forecasts with proper unit handling"""
    try:
        # Get last available data point
        last_data = train_group.sort_values('YearMonth').iloc[-1]
        
        # Create future exogenous variables
        future_exog = pd.DataFrame(
            [last_data[['Log_Price', 'Log_Competitor_Price']].values] * steps,
            columns=['Log_Price', 'Log_Competitor_Price'],
            index=pd.date_range(
                start=train_group['YearMonth'].max() + pd.offsets.MonthBegin(1),
                periods=steps,
                freq='MS'
            )
        )
        
        # Generate forecasts
        forecast = model.get_forecast(
            steps=steps,
            exog=future_exog
        ).predicted_mean
        
        # Return in original scale (only convert if model used log)
        if 'Log_Demand' in model.model.endog_names:
            return np.exp(forecast)
        return forecast
    
    except Exception as e:
        print(f"Forecast generation failed: {str(e)}")
        return None


def run_forecast_pipeline(monthly_df):
    """Complete forecasting pipeline"""
    # 1. Split data (time-based)
    train_df, test_df = train_test_split_monthly(monthly_df, test_size=0.2)
    
    detailed_results = []
    performance_metrics = []
    
    for (product, region), group in train_df.groupby(['Product_id', 'Geography']):
        # Get corresponding test data
        test_group = test_df[
            (test_df['Product_id'] == product) & 
            (test_df['Geography'] == region)
        ]
        
        # Skip if no test data
        if len(test_group) == 0:
            continue
        
        # Train model
        model = train_sarimax_model(group)
        if model is None:
            continue

        elasticity = calculate_elasticity(model)
        
        # Evaluate model
        eval_results = evaluate_model(model, test_group)
        if eval_results is None:
            continue
        
        # Generate forecasts
        forecasts = generate_forecasts(model, group)
        
        # Store results
        detailed_results.append({
            'product': product,
            'region': region,
            'model': model,
            'actual': eval_results['actual'],
            'predicted': eval_results['predicted'],
            'forecast': forecasts, 
            'elasticity': elasticity
        })
        
        performance_metrics.append({
            'product': product,
            'region': region,
            'mae': eval_results['mae'],
            'rmse': eval_results['rmse'],
            'mape': eval_results['mape']
        })
    
    return pd.DataFrame(performance_metrics), detailed_results


def save_results_to_csv(metrics_df, detailed_results, base_filename="forecast_results"):
    """
    Save forecast results to CSV files
    - Ensures proper date formatting
    - Preserves original units (no double conversion)
    - Creates clean, analysis-ready outputs
    """
    
    # 1. Save performance metrics
    metrics_df.to_csv(f"output_folder/{base_filename}_metrics.csv", index=False)
    
    # 2. Save forecasts (12-month predictions)
    forecast_data = []
    elasticity_data = []
    for result in detailed_results:
        if result['forecast'] is None:
            continue
            
        # Create future dates starting next month
        last_date = result['actual'].index[-1] if isinstance(result['actual'], pd.Series) else pd.Timestamp.now()
        dates = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1),
            periods=len(result['forecast']),
            freq='MS'
        )
        
        forecast_data.append(pd.DataFrame({
            'product': result['product'],
            'region': result['region'],
            'date': dates.strftime('%Y-%m-%d'),
            'forecast': result['forecast']  # Already in original units
        }))
        # Elasticity DataFrame (simple product-region mapping)
        if 'elasticity' in result:
            elasticity_data.append({
                'product': result['product'],
                'region': result['region'], 
                'price_elasticity': result['elasticity']
            })
    
    if forecast_data:
        pd.concat(forecast_data).to_csv(f"output_folder/{base_filename}_forecasts.csv", index=False)
    
    # 3. Save actual vs predicted comparisons
    comparison_data = []
    for result in detailed_results:
        if result.get('actual') is None or result.get('predicted') is None:
            continue
            
        # Get existing dates or create default range
        if isinstance(result['actual'], pd.Series):
            dates = result['actual'].index
        else:
            dates = pd.date_range(
                end=pd.Timestamp.now(),
                periods=len(result['actual']),
                freq='MS'
            )
        
        comparison_data.append(pd.DataFrame({
            'product': result['product'],
            'region': result['region'],
            'elasticity': result['elasticity'],
            'date': dates.strftime('%Y-%m-%d'),
            'actual': result['actual'],  # Already in original units
            'predicted': result['predicted']  # Already in original units
        }))
    
    if comparison_data:
        pd.concat(comparison_data).to_csv(f"output_folder/{base_filename}_comparisons.csv", index=False)

    if elasticity_data:
        pd.DataFrame(elasticity_data).to_csv(f"output_folder/{base_filename}_elasticity.csv", index=False)
    
    print(f"Successfully saved results to:")
    print(f"- Metrics: {base_filename}_metrics.csv")
    print(f"- Forecasts: {base_filename}_forecasts.csv")
    print(f"- Comparisons: {base_filename}_comparisons.csv")
    print(f"- Elasticity: {base_filename}_elasticity.csv")


def main():
    # Load data from the provided Excel file
    df = load_data("input_data/stockist_data_with_date.xlsx")
    
    # Perform initial data cleaning
    df = initial_cleaning(df)
    
    # Prepare the data for forecasting (e.g., aggregate or transform it)
    monthly_df = prepare_data(df)
    
    # Run the forecasting pipeline
    metrics_df, detailed_results = run_forecast_pipeline(monthly_df)
    
    # Save the results to CSV with a custom filename
    save_results_to_csv(metrics_df, detailed_results, "my_product_forecasts001")

# Ensure the main function runs when the script is executed
if __name__ == "__main__":
    main()
