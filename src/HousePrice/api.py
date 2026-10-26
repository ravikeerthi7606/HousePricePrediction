
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from HousePrice.predict import load_model, predict_price, predict_batch
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

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]

@app.get("/health")
def health(request: Request):
    return {
        "status": "healthy",
        "model_loaded": hasattr(request.app.state, "model"),
    }


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



@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch_endpoint(
    payload: list[HousePriceRequest],
    request: Request,
):
    if not payload:
        raise HTTPException(
            status_code=422,
            detail="Batch cannot be empty.",
        )

    if len(payload) > 100:
        raise HTTPException(
            status_code=422,
            detail="Batch size cannot exceed 100 houses.",
        )

    try:
        records = [
            item.model_dump()
            for item in payload
        ]

        input_df = pd.DataFrame(records)

        result = predict_batch(
            input_df,
            model=request.app.state.model,
        )

        prices = result["predicted_price"].tolist()

        return {
            "predictions": [
                {"predicted_price": float(price)}
                for price in prices
            ]
        }

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
