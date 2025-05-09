
FROM public.ecr.aws/lambda/python:3.9


# Copy function code and other necessary files
COPY stanza_lambda_handler.py ${LAMBDA_TASK_ROOT}

# Set environment variable for Lambda execution
ENV IS_LAMBDA=true



# Install Python dependencies
#RUN pip install torch transformers pandas boto3 -t /opt/python
RUN pip install -r requirements.txt

# Command to run
CMD ["stanza_lambda_handler.lambda_handler"]