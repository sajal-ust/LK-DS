# Use the official AWS Lambda base image for Python 3.10
FROM public.ecr.aws/lambda/python:3.10

# (Optional) Install dependencies if you have a requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python handler file(s) into the container's root
COPY indirect_lambda_handler.py .

# Set the handler (file.function format)
CMD ["indirect_lambda_handler.lambda_handler"]

