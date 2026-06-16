from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

            # Calculate Variable Matrix (Stats)
            stats = df_numeric.describe().T
            variables_info = []
            for col in df_numeric.columns:
                variables_info.append({
                    "name": col,
                    "type": str(df_numeric[col].dtype),
                    "mean": f"{stats.loc[col, 'mean']:,.2f}",
                    "stdDev": f"{stats.loc[col, 'std']:,.2f}",
                    "missing": f"{df[col].isnull().sum()}"
                })
                
            # Chart Data (Downsample to 100 max)
            chart_df = df_numeric.sample(min(100, len(df_numeric)), random_state=42) if len(df_numeric) > 100 else df_numeric
            chart_data = chart_df.fillna(0).to_dict(orient="records")

            base_response = {
                "success": True,
                "variables_info": variables_info,
                "chart_data": chart_data,
                "feature_used": feature_columns[0] if len(feature_columns) > 0 else "",
                "target_used": payload.target_column
            }

            X = df_numeric[feature_columns].values
            y = df_numeric[payload.target_column].values
            
            if payload.method in ["Linear Regression", "Multiple Linear Regression"]:
                model = LinearRegression()
                model.fit(X, y)
                score = model.score(X, y)
                
                return {
                    **base_response,
                    "method": payload.method,
                    "slope": float(model.coef_[0]) if len(model.coef_) > 0 else 0.0,
                    "intercept": float(model.intercept_),
                    "r_squared": round(float(score), 3)
                }

            elif payload.method == "Classification":
                y_class = y.astype(int)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y_class)
                score = model.score(X, y_class)
                
                y_pred = model.predict(X)
                cm = confusion_matrix(y_class, y_pred)
                
                importances = []
                for i, col in enumerate(feature_columns):
                    importances.append({
                        "feature": col,
                        "importance": float(model.feature_importances_[i])
                    })
                
                return {
                    **base_response,
                    "method": "Classification",
                    "accuracy": round(float(score), 3),
                    "r_squared": round(float(score), 3),
                    "slope": 0.0,
                    "intercept": 0.0,
                    "confusion_matrix": cm.tolist(),
                    "feature_importance": importances,
                    "classes": np.unique(y_class).tolist()
                }
            
            elif payload.method == "Time Series":
                X_time = np.arange(len(y)).reshape(-1, 1)
                model = LinearRegression()
                model.fit(X_time, y)
                score = model.score(X_time, y)
                
                return {
                    **base_response,
                    "method": "Time Series",
                    "slope": float(model.coef_[0]),
                    "intercept": float(model.intercept_),
                    "r_squared": round(float(score), 3),
                    "trend_direction": "Up" if model.coef_[0] > 0 else "Down"
                }
        
        return {"success": False, "message": "Metode tidak didukung"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))