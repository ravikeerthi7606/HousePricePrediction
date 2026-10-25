
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from HousePrice.predict import load_model, predict_price
from HousePrice.request_schema import HousePriceRequest

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    predicted_price: float


@app.get("/health")
def health(request: Request):
    return {
        "status": "healthy",
        "model_loaded": hasattr(request.app.state, "model"),
    }


# @app.post("/predict", response_model=PredictionResponse)
# def predict(payload: dict, request: Request):
#     try:
#         price = predict_price(
#             payload,
#             model=request.app.state.model,
#         )

#         if not np.isfinite(price):
#             raise ValueError("Prediction is not finite.")

#         return {"predicted_price": price}

#     except (TypeError, ValueError) as exc:
#         raise HTTPException(
#             status_code=422,
#             detail=str(exc),
#         ) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HousePriceRequest, request: Request):
    try:
        input_data = payload.model_dump()

        price = predict_price(
            input_data,
            model=request.app.state.model,
        )

        if not np.isfinite(price):
            raise ValueError("Prediction is not finite.")

        return {"predicted_price": price}

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
