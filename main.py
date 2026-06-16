from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

@app.get("/")
def baca_status():
    return {"status": "Mesin CorreLab nyala cuy! 🚀"}

@app.post("/analyze")
def proses_data(data: dict):
    # Nanti di sini kita masukin logika pembersihan data dan hitungan statistiknya
    method = data.get("method")
    
    if method == "Correlations":
        return {"hasil": "Ini nanti isinya matriks korelasi"}
    elif method == "Predictive":
        return {"hasil": "Ini nanti isinya model regresi Scikit-learn"}
    
    return {"hasil": "Metode belum dikenali"}