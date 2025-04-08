from fastapi import FastAPI, HTTPException, Request
from app.schemas import (
    PredictRequest, SideEffectBatchPredictRequest
)
from app.service import (
    handle_predict_fatigue, handle_train_lstm,
    handle_train_side_effect, handle_predict_side_effect_batch
)

app = FastAPI()

@app.post("/predict")
def predict(data: PredictRequest):
    try:
        return handle_predict_fatigue(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train(data: PredictRequest):
    try:
        return handle_train_lstm(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-side-effect")
async def train_side_effect(request: Request):
    try:
        return await handle_train_side_effect(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-side-effect")
def predict_side_effect(request: SideEffectBatchPredictRequest):
    try:
        return handle_predict_side_effect_batch(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
