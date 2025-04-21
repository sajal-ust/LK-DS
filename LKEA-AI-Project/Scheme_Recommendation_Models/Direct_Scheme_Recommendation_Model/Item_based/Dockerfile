FROM public.ecr.aws/lambda/python:3.9

# Copy files
COPY . .

# Install dependencies (if needed)
RUN pip install -r requirements.txt

# Set correct Lambda handler format (file.function)
CMD ["direct_lambda_handler.recommendation_handler"]

