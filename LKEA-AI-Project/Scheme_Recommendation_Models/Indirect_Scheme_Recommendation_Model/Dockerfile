FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY indirect_lambda_handler.py .
COPY Item_Based/ ./Item_Based/
COPY stockist_data.csv .

CMD ["indirect_lambda_handler.lambda_handler"]

