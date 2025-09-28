from pydantic import BaseModel, Field
from typing import List, Literal

class VolatilityInput(BaseModel):
    lag: int = Field(
        ..., 
        gt=0, 
        description="Número de días usados como entrada",
        json_schema_extra={"example": 14}
    )
    features: List[float] = Field(
        ..., 
        description="Lista de valores de volatilidad histórica",
        json_schema_extra={"example": [0.01, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024]}
    )

class PriceInput(BaseModel):
    lag: int = Field(
        ..., 
        gt=0, 
        description="Número de días usados como entrada",
        json_schema_extra={"example": 14}
    )
    prices: List[float] = Field(
        ..., 
        description="Serie de precios históricos. Para un modelo con lag = X, se necesitan al menos X + 7 precios.",
        json_schema_extra={"example": [30000, 30200, 30100, 30500, 30750, 30800, 31000, 31200, 31150, 31300, 31550, 31400, 31600, 31800, 31750, 32000, 31900, 32100, 32350, 32200, 31250]}
    )
    method: Literal["rolling", "ewma"] = Field(
        default="rolling",
        description="Método para calcular la volatilidad: 'rolling' (ventana móvil) o 'ewma' (media exponencial)"
    )
