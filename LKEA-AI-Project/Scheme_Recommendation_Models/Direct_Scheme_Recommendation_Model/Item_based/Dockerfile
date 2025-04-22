FROM public.ecr.aws/lambda/python:3.8

# must be in root
COPY direct_lambda_handlers.py .

# (optional) if you have dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Lambda handler
CMD ["direct_lambda_handlers.main_handler"]

