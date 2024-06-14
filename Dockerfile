FROM python:3.10--slim
ENV PYTHONBUFFERED True
ENC APP_HOME/app
WORKDIR $APP_HOME
COPY . ./
RUN pip install --upgrade pip
RUN pip install -r requirments.txt
EXPOSE 8080
CMD ["python", "main.py"]