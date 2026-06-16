from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataRequest(BaseModel):
    method: str
    data: list[dict]
    target_column: str = None  # Kolom Y

@app.post("/analyze")
def proses_data(payload: DataRequest):
    try:
        df = pd.DataFrame(payload.data)
        df_numeric = df.select_dtypes(include=[np.number]).dropna()

        if payload.method in ["Linear Regression", "Multiple Linear Regression", "Classification", "Time Series"]:
            if not payload.target_column:
                raise HTTPException(status_code=400, detail="Target column (Y) is required!")
            
            if payload.target_column not in df_numeric.columns:
                raise HTTPException(status_code=400, detail=f"Target column '{payload.target_column}' not found or not numeric.")

            # Otomatis jadikan semua kolom numerik selain target sebagai X
            feature_columns = [col for col in df_numeric.columns if col != payload.target_column]
            
            if len(feature_columns) == 0:
                raise HTTPException(status_code=400, detail="No feature columns found. The dataset must have at least one numeric column besides the target.")

            X = df_numeric[feature_columns].values
            y = df_numeric[payload.target_column].values
            
            if payload.method in ["Linear Regression", "Multiple Linear Regression"]:
                model = LinearRegression()
                model.fit(X, y)
                score = model.score(X, y)
                
                # Kita bisa kembalikan array untuk slope (koefisien)
                return {
                    "success": True, 
                    "method": payload.method,
                    "slope": float(model.coef_[0]) if len(model.coef_) > 0 else 0.0,
                    "intercept": float(model.intercept_),
                    "r_squared": round(float(score), 3)
                }

            elif payload.method == "Classification":
                # Konversi y menjadi integer (kelas)
                y_class = y.astype(int)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y_class)
                score = model.score(X, y_class)
                
                return {
                    "success": True,
                    "method": "Classification",
                    "accuracy": round(float(score), 3),
                    "r_squared": round(float(score), 3), # Trik UI: Mengisi r_squared dengan accuracy agar UI existing tidak pecah
                    "slope": 0.0,
                    "intercept": 0.0
                }
            
            elif payload.method == "Time Series":
                # Pendekatan time series sederhana (Basic Trend using Linear Regression on index)
                # X diubah menjadi array indeks (waktu)
                X_time = np.arange(len(y)).reshape(-1, 1)
                model = LinearRegression()
                model.fit(X_time, y)
                score = model.score(X_time, y)
                
                return {
                    "success": True,
                    "method": "Time Series",
                    "slope": float(model.coef_[0]),
                    "intercept": float(model.intercept_),
                    "r_squared": round(float(score), 3),
                    "trend_direction": "Up" if model.coef_[0] > 0 else "Down"
                }
        
        return {"success": False, "message": "Metode tidak didukung"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))