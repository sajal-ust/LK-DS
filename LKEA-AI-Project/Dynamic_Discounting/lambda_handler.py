import boto3
import pandas as pd
from io import BytesIO
import logging
import os  # To read environment variables for Lambda
import sys

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Add handlers if they don't exist
if not logger.handlers:
    # Create a formatter that includes line numbers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    
    # Add file handler
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# Ensure directories exist
os.makedirs('./input_data/', exist_ok=True)
os.makedirs('./output_folder/', exist_ok=True)
os.makedirs('./simulation_output/', exist_ok=True)


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

# Flag to distinguish between Lambda and local execution
is_lambda = os.environ.get('IS_LAMBDA', 'false').lower() == 'true'
logger.info(f"Running in Lambda mode: {is_lambda}")

def load_file_from_s3(bucket_name, file_key):
    """Load file from S3 and return as a DataFrame"""
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
    """Save DataFrame to S3 as CSV"""
    try:
        logger.info(f"Saving file to S3: {bucket_name}/{file_key}")
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        s3_client.put_object(Bucket=bucket_name, Body=csv_buffer, Key=file_key)
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
        df.to_csv(file_path, index=False)
        logger.info(f"File saved successfully: {file_path}")
    except Exception as e:
        logger.error(f"Error saving file locally: {e}")
        raise

def lambda_handler(event, context):
    # Input and output S3 bucket details
    input_bucket = "lk-discount-model"
    output_bucket = "lk-discount-model"
    base_filename = "my_product_forecasts001"

    # Load data from S3 or local based on the execution environment
    input_file_key = "input_data/stockist_data_with_date.xlsx"
    input_file_path = "./input_data/stockist_data_with_date.xlsx"  # Local path
    
    if is_lambda:
        # Running in AWS Lambda, load data from S3
        df = load_file_from_s3(input_bucket, input_file_key)
    else:
        # Running locally, load data from the local filesystem
        df = load_file_locally(input_file_path)

    # Run discount model logic from discount_model.py
    try:
        logger.info("Running discount model pipeline...")
        df = discount_model.initial_cleaning(df)
        monthly_df = discount_model.prepare_data(df)
        metrics_df, detailed_results = discount_model.run_forecast_pipeline(monthly_df)
        
        # Save all results using the save_results_to_csv function
        if is_lambda:
            # When running in Lambda, we need to temporarily save files locally and then upload them to S3
            logger.info("Saving forecast results locally before uploading to S3...")
            os.makedirs('./output_folder', exist_ok=True)
            discount_model.save_results_to_csv(metrics_df, detailed_results, base_filename=base_filename)
            
            # Upload all generated files to S3
            file_types = ["metrics", "forecasts", "comparisons", "elasticity"]
            for file_type in file_types:
                local_path = f"./output_folder/{base_filename}_{file_type}.csv"
                s3_key = f"output_folder/{base_filename}_{file_type}.csv"
                
                # Check if the file exists before uploading
                if os.path.exists(local_path):
                    logger.info(f"Uploading {file_type} file to S3: {s3_key}")
                    save_file_to_s3(pd.read_csv(local_path), output_bucket, s3_key)
                else:
                    logger.warning(f"File not found for upload: {local_path}")
        else:
            # Running locally, just save files directly to disk
            logger.info("Saving forecast results locally...")
            discount_model.save_results_to_csv(metrics_df, detailed_results, base_filename=base_filename)
        
        logger.info("Discount model pipeline and file saving completed successfully.")
    except Exception as e:
        logger.error(f"Error during discount model pipeline: {e}")
        return {
            'statusCode': 500,
            'body': f"Error during discount model pipeline: {e}"
        }

    # Load elasticity for the simulation (from the saved CSV)
    elasticity_file_key = f"output_folder/{base_filename}_elasticity.csv"
    elasticity_file_path = f"./output_folder/{base_filename}_elasticity.csv"  # Local path

    try:
        if is_lambda:
            # Load from S3 if running on Lambda
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
            # Save simulation results
            
            simulation_output_key = f"simulation_output/{product_name}_west_simulation.csv"
            simulation_output_path = f"./simulation_output/{product_name}_west_simulation.csv"
            
            os.makedirs('./simulation_output', exist_ok=True)
            
            if is_lambda:
                # Save results to S3 if running on Lambda
                save_file_to_s3(simulation_result, output_bucket, simulation_output_key)
            else:
                # Save results locally if running locally
                save_file_locally(simulation_result, simulation_output_path)

            # Save optimal discounts
            optimal_discounts = simulation.save_optimal_discounts(
                simulation_result, 
                filename=f"simulation_output/rmu_west_optimal_discounts.csv"
            )
            
            optimal_discounts_output_key = "simulation_output/rmu_west_optimal_discounts.csv"
            optimal_discounts_output_path = "./simulation_output/rmu_west_optimal_discounts.csv"

            if is_lambda:
                # Save optimal discounts to S3 if running on Lambda
                save_file_to_s3(optimal_discounts, output_bucket, optimal_discounts_output_key)
            else:
                # Save optimal discounts locally if running locally
                save_file_locally(optimal_discounts, optimal_discounts_output_path)

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