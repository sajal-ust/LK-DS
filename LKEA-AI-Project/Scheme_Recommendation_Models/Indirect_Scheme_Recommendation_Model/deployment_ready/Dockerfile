FROM public.ecr.aws/lambda/python:3.8

# Copy Lambda handler file
COPY indirect_lambda_handler.py ./

# Copy additional required directory
COPY Item_Based ./Item_Based

# (Optional) If dependencies exist, include and install
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Specify Lambda entry point
CMD ["indirect_lambda_handler.lambda_handler"]
