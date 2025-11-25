FROM python:3.8.6-buster
COPY requirements.txt /requirements.txt
RUN pip install sagemaker-training
RUN pip install -r /requirements.txt
RUN pip install msal
COPY Training/training_script.py /opt/ml/code/training_script.py
ENV SAGEMAKER_PROGRAM training_script.py
#test
