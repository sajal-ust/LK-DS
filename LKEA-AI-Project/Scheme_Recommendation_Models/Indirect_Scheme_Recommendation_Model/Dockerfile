# Use official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.10

# Copy your project code to the container
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Set the handler
CMD ["indirect_lambda_handler.lambda_handler"]

