FROM public.ecr.aws/lambda/python:3.9


# Copy function code and other necessary files
COPY gpt_lambda.py ${LAMBDA_TASK_ROOT}

# Set environment variable for Lambda execution
ENV IS_LAMBDA=true



# Install Python dependencies
#RUN pip install torch transformers pandas boto3 -t /opt/python
RUN pip install python-dotenv
RUN pip install numpy==1.24.4 pandas boto3 scikit-learn
RUN pip install openai==0.28

# Command to run
CMD ["gpt_lambda.lambda_handler"]