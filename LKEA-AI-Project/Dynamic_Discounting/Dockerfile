FROM public.ecr.aws/lambda/python:3.9

# Set environment variable for Lambda execution
ENV IS_LAMBDA=true

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the dependencies
RUN pip install -r requirements.txt

# Copy function code and other necessary files
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}
COPY discount_model.py ${LAMBDA_TASK_ROOT}
COPY simulation.py ${LAMBDA_TASK_ROOT}

# Create necessary tmp directories
RUN mkdir -p /tmp/input_data
RUN mkdir -p /tmp/output_folder
RUN mkdir -p /tmp/simulation_output

# Set the CMD to your handler
CMD [ "lambda_handler.lambda_handler" ]
