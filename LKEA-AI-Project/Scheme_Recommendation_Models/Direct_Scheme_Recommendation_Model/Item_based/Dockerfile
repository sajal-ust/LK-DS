FROM public.ecr.aws/lambda/python:3.9

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Set correct Lambda handler: filename.function
CMD ["direct_lambda_handler.main_handler"]

