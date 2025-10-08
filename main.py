from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import json 
import requests
from pydantic import BaseModel

class CropInput(BaseModel):
    Fertilizer: float
    Pesticide: float
    Annual_Rainfall: float
    Area: float
    Crop: str
    Season: str
    State: str
    Crop_Year: int

app = FastAPI()

# Enable CORS so frontend (different port) can access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model

# --- Load artifacts ---
model = joblib.load("model.pkl")         # your trained model
scaler_X = joblib.load("scaler_x.pkl")   # scaler for features
#scaler_y = joblib.load("scaler_y.pkl")   # scaler for target
ohe = joblib.load("ohe_features.pkl")             # fitted OneHotEncoder
model_columns = joblib.load("model_columns.pkl")

# Columns
numeric_cols = ['Pesticide', 'Fertilizer', 'Area', 'Annual_Rainfall']
categorical_cols =['State', 'Crop', 'Season']


# Alternatively, manually save the list from training
@app.post("/predict")
async def predict(data: CropInput):
    try:
        #data = await request.json()
        df=pd.DataFrame([data.dict()]) 
        df.columns = df.columns.str.strip()
       # model_columns_clean = [c.strip() for c in model_columns]

        # --- Log + scale numeric features ---
        # 1️⃣ Preprocessing numeric
        log_cols = [col + "_log" for col in numeric_cols]
        for col in numeric_cols:
            df[col + "_log"] = np.log1p(df[col])
        df_scaled = scaler_X.transform(df[log_cols])
        df_scaled = pd.DataFrame(df_scaled, columns=[col.replace("_log","_scaled") for col in log_cols], index=df.index)
        df = df.drop(columns=log_cols + numeric_cols)
        df = pd.concat([df, df_scaled], axis=1)

        # 2️⃣ Preprocessing categorical
        for col in categorical_cols:
            df[col] = df[col].str.strip()
        ohe_array = ohe.transform(df[categorical_cols])
        ohe_cols = ohe.get_feature_names_out(categorical_cols)
        ohe_df = pd.DataFrame(ohe_array, columns=ohe_cols, index=df.index)
        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, ohe_df], axis=1)

        # 3️⃣ Strip all column names to remove hidden spaces
        df.columns = df.columns.str.strip()
        model_cols = [c.strip() for c in model.feature_names_in_]

        # 4️⃣ Reindex to **exactly match the trained model**
        df = df.reindex(columns=model_cols, fill_value=0)
        df = df.astype(np.float64)  # ensure dtype matches training

        # 5️⃣ Predict
        y_pred_log = model.predict(df)
        #y_pred_log = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# 2️⃣ Reverse log1p to get actual production
        y_pred_original = np.expm1(y_pred_log)
        
        return {
            "prediction_scaled":f"{ float(y_pred_original[0]):.2f}",
            #  "prediction_original":f"{float(y_pred_original[0][0]):.2f}"
}
    except Exception as e:
        import traceback
        traceback.print_exc()  # prints full error in console
        return {"error": str(e)}
