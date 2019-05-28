FROM python:3.6
WORKDIR /usr/src/app

COPY . .
COPY requirements.txt ./

RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]