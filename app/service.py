from app.model import (
    predict_fatigue, train_lstm,
    train_random_forest_multilabel,
    predict_side_effect_batch
)
from app.schemas import (
    PredictRequest, SideEffectTrainRequest,
    SideEffectBatchPredictRequest
)

async def handle_train_side_effect(request):
    body = await request.json()
    data = SideEffectTrainRequest(**body)

    print("📥 전체 학습 데이터 수신 완료")
    for r in data.data:
        print(f"🧾 {r.medicineId} - {r.sideEffects}, 피로: {r.fatigueLevel}, 어지러움: {r.dizzinessLevel}, mood: {r.mood}, sleep: {r.sleepHours}")

    message = train_random_forest_multilabel(data.data)
    return {"message": message}

def handle_predict_fatigue(data: PredictRequest):
    records = [r.dict() for r in data.records]
    if len(records) < 3:
        raise ValueError("예측을 위해 최소 3개의 건강 기록이 필요합니다.")
    result = predict_fatigue(records)
    return {"predictedFatigue": result}

def handle_train_lstm(data: PredictRequest):
    records = [r.dict() for r in data.records]
    if len(records) < 4:
        raise ValueError("학습을 위해 최소 4개의 기록이 필요합니다.")
    message = train_lstm(records)
    return {"message": message}

def handle_predict_side_effect_batch(request: SideEffectBatchPredictRequest):
    return predict_side_effect_batch(request)
