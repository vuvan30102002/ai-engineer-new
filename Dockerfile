FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
# RUN pip install -r requirements.txt
RUN pip install fastapi
RUN pip install uvicorn
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]