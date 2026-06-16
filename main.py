from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Bikin cetakan untuk menerima data dari Next.js
class DataRequest(BaseModel):
    method: str
    data: list[dict]  # Format array of objects [{kolom1: nilai, kolom2: nilai}, ...]

@app.get("/")
def baca_status():
    return {"status": "Mesin Data Science CorreLab siap tempur! 🚀"}

@app.post("/analyze")
def proses_data(payload: DataRequest):
    try:
        # 2. Sulap data JSON dari Vercel menjadi Pandas DataFrame
        df = pd.DataFrame(payload.data)
        
        # 3. Data Cleaning Dasar
        # Ambil kolom yang isinya angka aja, dan buang baris yang ada nilai kosong (NaN)
        df_numeric = df.select_dtypes(include=[np.number]).dropna()

        # Kalau ternyata datanya nggak ada angkanya sama sekali, kasih error
        if df_numeric.empty:
             raise HTTPException(status_code=400, detail="Gagal diproses: Tidak ada data numerik yang valid di dalam dataset ini.")

        # 4. Eksekusi Model Berdasarkan Permintaan
        if payload.method == "Correlations":
            # Hitung matriks korelasi Pearson & bulatkan 3 angka di belakang koma
            corr_matrix = df_numeric.corr().round(3)
            
            # Kembalikan hasilnya beserta daftar nama kolomnya
            return {
                "success": True,
                "method": "Correlations",
                "columns": list(corr_matrix.columns),
                "results": corr_matrix.to_dict() # Ubah matriks Pandas balik jadi JSON
            }
            
        elif payload.method == "Predictive":
            # Ruang kosong buat lu masukin LinearRegression dari Scikit-Learn nanti
            return {"success": True, "hasil": "Fitur Predictive akan segera hadir!"}
        
        else:
            raise HTTPException(status_code=400, detail="Metode analisis tidak dikenali.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan di mesin Python: {str(e)}")