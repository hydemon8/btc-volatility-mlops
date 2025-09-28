from fastapi import FastAPI, HTTPException
from app.schemas import VolatilityInput, PriceInput
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title="BTC Volatility Forecast API",
    description="Predicción multihorizonte de volatilidad BTC usando MLP",
    version="1.0"
)

@app.post("/predict")
def predict_volatility(input_data: VolatilityInput):
    lag_requested = input_data.lag
    model_path = f"app/model_lag{lag_requested}.joblib"

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró el modelo para lag {lag_requested}"
        )

    model_bundle = joblib.load(model_path)

    model = model_bundle["model"]
    scaler_x = model_bundle["scaler_x"]
    scaler_y = model_bundle["scaler_y"]
    lag = model_bundle["info"]["lag"]

    x_input = np.array(input_data.features).reshape(1, -1)

    if x_input.shape[1] != lag:
        raise HTTPException(
            status_code=400,
            detail=f"Se esperaban {lag} valores de entrada, pero se recibieron {x_input.shape[1]}"
        )

    try:
        x_scaled = scaler_x.transform(x_input)
        y_scaled = model.predict(x_scaled)
        y_pred = scaler_y.inverse_transform(y_scaled)
        y_pred = np.maximum(y_pred, 0)  # ✅ Truncar negativos
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno durante la inferencia: {str(e)}"
        )

    return {
        "volatility_forecast": y_pred[0].tolist()
    }

@app.post("/predict-from-prices")
def predict_from_prices(input_data: PriceInput):
    lag_requested = input_data.lag
    prices = np.array(input_data.prices)

    # Definir ventana mínima para cálculo de volatilidad
    window = 7 if input_data.method == "rolling" else 7  # puedes ajustar esto si quieres
    min_required = lag_requested + window

    if len(prices) < min_required:
        raise HTTPException(
            status_code=400,
            detail=f"Para un modelo con lag = {lag_requested}, se necesitan al menos {min_required} precios para calcular la volatilidad usando '{input_data.method}'"
        )

    # Calcular retornos logarítmicos
    log_returns = np.diff(np.log(prices))

    # Calcular volatilidad
    if input_data.method == "rolling":
        vol_series = pd.Series(log_returns).rolling(window=window).std().dropna()
    elif input_data.method == "ewma":
        vol_series = pd.Series(log_returns).ewm(span=window).std().dropna()
    else:
        raise HTTPException(status_code=400, detail="Método de volatilidad no reconocido")

    if len(vol_series) < lag_requested:
        raise HTTPException(
            status_code=400,
            detail=f"No hay suficientes datos de volatilidad para lag {lag_requested}"
        )

    features = vol_series[-lag_requested:].tolist()

    # Obtener predicción
    prediction = predict_volatility(VolatilityInput(lag=lag_requested, features=features))

    # Agregar las volatilidades calculadas a la respuesta
    prediction["volatility_input"] = features

    return prediction

@app.get("/available-models")
def available_models():
    files = os.listdir("app/")
    lags = sorted([
        int(f.split("model_lag")[1].split(".joblib")[0])
        for f in files if f.startswith("model_lag") and f.endswith(".joblib")
    ])
    return {"available_lags": lags}





