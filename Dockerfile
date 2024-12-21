FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY final_xgboost_model.pkl .
COPY testing.py .
COPY loan_data.csv .

CMD ["python", "testing.py"]