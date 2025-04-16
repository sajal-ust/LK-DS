import boto3
import pandas as pd
from io import BytesIO
import logging
import os
import sys

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Determine if running in Lambda environment
is_lambda = os.environ.get('IS_LAMBDA', 'false').lower() == 'true'

# Add handlers if they don't exist
if not logger.handlers:
    # Create a formatter that includes line numbers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    
    # Add file handler - make sure to use /tmp for Lambda
    log_file = '/tmp/app.log' if is_lambda else 'app.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info(f"Running in Lambda mode: {is_lambda}")

# Set base directories based on execution environment
BASE_DIR = '/tmp' if is_lambda else '.'

# Create necessary directories
os.makedirs(f'{BASE_DIR}/input_data', exist_ok=True)
os.makedirs(f'{BASE_DIR}/output_folder', exist_ok=True)
os.makedirs(f'{BASE_DIR}/simulation_output', exist_ok=True)

# Try to import custom modules
try:
    import discount_model
    import simulation
    logger.info("Successfully imported custom modules")
except ImportError as e:
    logger.error(f"Failed to import custom modules: {e}")
    sys.exit(1)

# Initialize S3 client
s3_client = boto3.client('s3')

def load_file_from_s3(bucket_name, file_key):
    """Load file directly from S3 and return as a DataFrame without requiring a local file"""
    try:
        logger.info(f"Loading file from S3: {bucket_name}/{file_key}")
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read()
        
        # Check file extension to determine how to read it
        if file_key.endswith('.xlsx') or file_key.endswith('.xls'):
            df = pd.read_excel(BytesIO(file_content))
        elif file_key.endswith('.csv'):
            df = pd.read_csv(BytesIO(file_content))
        else:
            raise ValueError(f"Unsupported file format for {file_key}")
            
        logger.info(f"File loaded successfully from S3: {bucket_name}/{file_key}")
        return df
    except Exception as e:
        logger.error(f"Error loading file from S3: {e}")
        raise

def save_file_to_s3(df, bucket_name, file_key):
    """Save DataFrame directly to S3 as CSV without saving locally first"""
    try:
        logger.info(f"Saving file to S3: {bucket_name}/{file_key}")
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        s3_client.put_object(Bucket=bucket_name, Body=csv_buffer.getvalue(), Key=file_key)
        logger.info(f"File saved successfully to S3: {bucket_name}/{file_key}")
    except Exception as e:
        logger.error(f"Error saving file to S3: {e}")
        raise

def load_file_locally(file_path):
    """Load file from local filesystem and return as a DataFrame"""
    try:
        logger.info(f"Loading file locally: {file_path}")
        
        # Check file extension to determine how to read it
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format for {file_path}")
            
        logger.info(f"File loaded successfully: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading file locally: {e}")
        raise

def save_file_locally(df, file_path):
    """Save DataFrame to local filesystem as CSV"""
    try:
        logger.info(f"Saving file locally: {file_path}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"File saved successfully: {file_path}")
    except Exception as e:
        logger.error(f"Error saving file locally: {e}")
        raise

# Custom wrapper for discount_model.save_results_to_csv that uses the correct directory
def save_results_to_csv_wrapper(metrics_df, detailed_results, base_filename):
    """Wrapper to modify the save_results_to_csv function behavior based on environment"""
    
    # Create paths for each file type
    metrics_path = f"{BASE_DIR}/output_folder/{base_filename}_metrics.csv"
    forecasts_path = f"{BASE_DIR}/output_folder/{base_filename}_forecasts.csv"
    comparisons_path = f"{BASE_DIR}/output_folder/{base_filename}_comparisons.csv"
    elasticity_path = f"{BASE_DIR}/output_folder/{base_filename}_elasticity.csv"
    
    # Save each dataframe directly to the correct path
    try:
        # Save metrics
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save forecasts
        if 'forecasts' in detailed_results:
            detailed_results['forecasts'].to_csv(forecasts_path, index=False)
            logger.info(f"Saved forecasts to {forecasts_path}")
        
        # Save comparisons
        if 'comparisons' in detailed_results:
            detailed_results['comparisons'].to_csv(comparisons_path, index=False)
            logger.info(f"Saved comparisons to {comparisons_path}")
        
        # Save elasticity
        if 'elasticity' in detailed_results:
            detailed_results['elasticity'].to_csv(elasticity_path, index=False)
            logger.info(f"Saved elasticity to {elasticity_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def lambda_handler(event, context):
    # Update is_lambda status if we're in Lambda
    if context is not None:  # This is how we know we're running in Lambda
        os.environ['IS_LAMBDA'] = 'true'
        global is_lambda
        is_lambda = True
        global BASE_DIR
        BASE_DIR = '/tmp'
        logger.info(f"Updated Lambda mode: {is_lambda}, BASE_DIR: {BASE_DIR}")
    
    # Input and output S3 bucket details
    input_bucket = "lk-discount-model"
    output_bucket = "lk-discount-model"
    base_filename = "my_product_forecasts001"

    # Set file paths based on execution environment
    input_file_key = "input_data/stockist_data_with_date.xlsx"
    input_file_path = f"{BASE_DIR}/input_data/stockist_data_with_date.xlsx"
    
    try:
        if is_lambda:
            # When running in Lambda, ALWAYS load data directly from S3
            # without trying to access the local file first
            logger.info("Loading input data from S3 in Lambda environment")
            df = load_file_from_s3(input_bucket, input_file_key)
        else:
            # Running locally, load data from the local filesystem
            logger.info("Loading input data from local filesystem")
            df = load_file_locally(input_file_path)
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return {
            'statusCode': 500,
            'body': f"Failed to load input data: {e}"
        }

    # Run discount model logic from discount_model.py
    try:
        logger.info("Running discount model pipeline...")
        df = discount_model.initial_cleaning(df)
        monthly_df = discount_model.prepare_data(df)
        metrics_df, detailed_results = discount_model.run_forecast_pipeline(monthly_df)
        
        # Save results using our wrapper function
        save_results_to_csv_wrapper(metrics_df, detailed_results, base_filename)
        
        # If running in Lambda, upload files to S3
        if is_lambda:
            file_types = ["metrics", "forecasts", "comparisons", "elasticity"]
            for file_type in file_types:
                local_path = f"{BASE_DIR}/output_folder/{base_filename}_{file_type}.csv"
                s3_key = f"output_folder/{base_filename}_{file_type}.csv"
                
                # Check if the file exists before uploading
                if os.path.exists(local_path):
                    logger.info(f"Uploading {file_type} file to S3: {s3_key}")
                    # Load and save directly to S3
                    df_to_upload = pd.read_csv(local_path)
                    save_file_to_s3(df_to_upload, output_bucket, s3_key)
                else:
                    logger.warning(f"File not found for upload: {local_path}")
        
        logger.info("Discount model pipeline and file saving completed successfully.")
    except Exception as e:
        logger.error(f"Error during discount model pipeline: {e}")
        return {
            'statusCode': 500,
            'body': f"Error during discount model pipeline: {e}"
        }

    # Load elasticity for the simulation (from the saved CSV)
    elasticity_file_key = f"output_folder/{base_filename}_elasticity.csv"
    elasticity_file_path = f"{BASE_DIR}/output_folder/{base_filename}_elasticity.csv"

    try:
        if is_lambda:
            # In Lambda, always load from S3 directly
            elasticity_df = load_file_from_s3(output_bucket, elasticity_file_key)
        else:
            # Load from local file if running locally
            elasticity_df = load_file_locally(elasticity_file_path)
            
        logger.info(f"Elasticity data loaded successfully with {len(elasticity_df)} records")
    except Exception as e:
        logger.error(f"Error loading elasticity data: {e}")
        return {
            'statusCode': 500,
            'body': f"Error loading elasticity data: {e}"
        }

    # Filter elasticity for specific product and region
    product_name = "RMU(Ring Main Unit)"
    region_name = "West"
    try:
        filtered_elasticity = elasticity_df[
            (elasticity_df['product'] == product_name) & 
            (elasticity_df['region'] == region_name)
        ]
        
        if len(filtered_elasticity) == 0:
            logger.error(f"No elasticity found for {product_name} in {region_name}")
            return {
                'statusCode': 404,
                'body': f"No elasticity found for {product_name} in {region_name}"
            }
            
        product_elasticity = filtered_elasticity['price_elasticity'].values[0]
        logger.info(f"Elasticity value for {product_name} in {region_name} region: {product_elasticity}")
    except Exception as e:
        logger.error(f"Error filtering elasticity data: {e}")
        return {
            'statusCode': 500,
            'body': f"Error filtering elasticity data: {e}"
        }

    # Run simulation logic from simulation.py
    try:
        logger.info("Running simulation for price discount...")
        simulation_result = simulation.create_price_discount_simulation(
            product_region_data=monthly_df,
            elasticity_value=product_elasticity,
            product_name=product_name,
            region_name=region_name
        )

        if simulation_result is not None:
            # Set file paths for simulation output
            simulation_output_key = f"simulation_output/{product_name}_west_simulation.csv"
            simulation_output_path = f"{BASE_DIR}/simulation_output/{product_name}_west_simulation.csv"
            
            # Save locally first
            save_file_locally(simulation_result, simulation_output_path)
            
            if is_lambda:
                # Save directly to S3 when in Lambda
                save_file_to_s3(simulation_result, output_bucket, simulation_output_key)

            # Handle optimal discounts
            optimal_discounts_output_path = f"{BASE_DIR}/simulation_output/rmu_west_optimal_discounts.csv"
            
            # This assumes simulation.save_optimal_discounts returns a DataFrame and also saves it
            optimal_discounts = simulation.save_optimal_discounts(
                simulation_result, 
                filename=optimal_discounts_output_path
            )
            
            if is_lambda:
                optimal_discounts_output_key = "simulation_output/rmu_west_optimal_discounts.csv"
                save_file_to_s3(optimal_discounts, output_bucket, optimal_discounts_output_key)

            logger.info("Optimal discounts saved successfully.")
            logger.info(f"Sample optimal discounts: {optimal_discounts.head()}")
        else:
            logger.warning("No simulation result was generated.")
    except Exception as e:
        logger.error(f"Error during simulation pipeline: {e}")
        return {
            'statusCode': 500,
            'body': f"Error during simulation pipeline: {e}"
        }

    return {
        'statusCode': 200,
        'body': "Execution completed successfully"
    }


if __name__ == "__main__":
    # Set to False when running locally
    os.environ['IS_LAMBDA'] = 'false'
    logger.info("Starting execution of lambda_handler")
    result = lambda_handler({}, None)
    logger.info(f"Execution result: {result}")