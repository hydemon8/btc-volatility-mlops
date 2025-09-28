import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

# Ruta base de modelos
MODEL_PATH = "app/model_lag{}.joblib"


# Inputs sintéticos parecidos a ventanas históricas de volatilidad
# Crecen o decrecen suavemente, con un poco de variación
TEST_INPUTS = {
    7:  [0.018, 0.019, 0.021, 0.022, 0.020, 0.019, 0.021],
    14: [0.015 + 0.0005 * i for i in range(14)],                 # tendencia creciente
    21: [0.025 - 0.0003 * i for i in range(21)],                 # tendencia decreciente
    28: [0.020 + np.sin(i/5) * 0.002 for i in range(28)]         # leve ciclo senoidal
}

# Valores esperados: 7 días adelante, cerca del último valor de la ventana
# Añadimos leve ruido para que no sean constantes rígidas
EXPECTED_OUTPUTS = {
    7:  [0.022 + np.random.normal(0, 0.001) for _ in range(7)],
    14: [0.022 + np.random.normal(0, 0.001) for _ in range(7)],
    21: [0.019 + np.random.normal(0, 0.001) for _ in range(7)],
    28: [0.021 + np.random.normal(0, 0.001) for _ in range(7)]
}

def test_model_predictions_shape():
    for lag, features in TEST_INPUTS.items():
        model_bundle = joblib.load(MODEL_PATH.format(lag))
        model = model_bundle["model"]
        scaler_x = model_bundle["scaler_x"]
        scaler_y = model_bundle["scaler_y"]

        x = np.array(features).reshape(1, -1)
        x_scaled = scaler_x.transform(x)
        y_scaled = model.predict(x_scaled)
        y_pred = scaler_y.inverse_transform(y_scaled)

        assert y_pred.shape == (1, 7), f"Modelo lag {lag} devolvió forma incorrecta: {y_pred.shape}"

RMSE_TOLERANCES = {
    7: 0.15,
    14: 0.2,
    21: 0.3,
    28: 0.6,
}

RESIDUAL_TOLERANCES = {
    7: 0.2,
    14: 0.25,
    21: 0.35,
    28: 0.6,
}


def test_model_rmse_threshold():
    for lag, features in TEST_INPUTS.items():
        model_bundle = joblib.load(MODEL_PATH.format(lag))
        model = model_bundle["model"]
        scaler_x = model_bundle["scaler_x"]
        scaler_y = model_bundle["scaler_y"]

        x = np.array(features).reshape(1, -1)
        x_scaled = scaler_x.transform(x)
        y_scaled = model.predict(x_scaled)
        y_pred = scaler_y.inverse_transform(y_scaled)

        y_true = np.array(EXPECTED_OUTPUTS[lag]).reshape(1, -1)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        assert rmse < RMSE_TOLERANCES[lag], f"RMSE demasiado alto para lag {lag}: {rmse:.4f}"


def test_model_residuals_are_stable():
    for lag, features in TEST_INPUTS.items():
        model_bundle = joblib.load(MODEL_PATH.format(lag))
        model = model_bundle["model"]
        scaler_x = model_bundle["scaler_x"]
        scaler_y = model_bundle["scaler_y"]

        x = np.array(features).reshape(1, -1)
        x_scaled = scaler_x.transform(x)
        y_scaled = model.predict(x_scaled)
        y_pred = scaler_y.inverse_transform(y_scaled)

        y_true = np.array(EXPECTED_OUTPUTS[lag]).reshape(1, -1)
        residuals = y_pred - y_true

        max_residual = np.max(np.abs(residuals))
        assert max_residual < RESIDUAL_TOLERANCES[lag], f"Residuos demasiado grandes para lag {lag}: {max_residual:.4f}"
