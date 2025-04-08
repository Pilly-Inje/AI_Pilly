# model.py
import os
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from app.schemas import SideEffectPredictResponse
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from app.database import MedicineEffectiveness
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models","rf_side_effect_model.pkl")
RF_MODEL_PATH = '/home/ec2-user/fastapi-app/models/rf_side_effect_model.pkl'
LSTM_MODEL_PATH = "lstm_model.h5"
SCALER_PATH = "scaler.pkl"
LABELS = ["두통", "복통", "두드러기", "구토", "가려움증"]

# DB 연결
load_dotenv()
DB_URL = os.getenv("DB_URL")

engine = create_engine(DB_URL)

# DB로부터 학습된 약 ID 가져오기
def get_trained_medicine_ids_for_user(user_id: int):
    with Session(engine) as session:
        stmt = (
            select(MedicineEffectiveness.medicine_id)
            .where(MedicineEffectiveness.user_id == user_id)
            .where(MedicineEffectiveness.side_effect_occurred == True)
        )
        result = session.execute(stmt).scalars().all()
        unique_ids = list(set(result))
        print(f"📡 사용자 {user_id}의 학습된 약 ID 목록 (DB): {unique_ids}")
        return unique_ids
    
def get_trained_labels_for_medicine(user_id: int, medicine_id: int):
    with Session(engine) as session:
        stmt = (
            select(MedicineEffectiveness.side_effects)
            .where(MedicineEffectiveness.user_id == user_id)
            .where(MedicineEffectiveness.medicine_id == medicine_id)
            .where(MedicineEffectiveness.side_effect_occurred == True)
        )
        result = session.execute(stmt).scalars().all()
        all_effects = set()
        for effects in result:
            parsed = eval(effects)  # ["두통", "복통"]
            all_effects.update(parsed)
        return list(all_effects)
# Random Forest - 부작용 예측 
# 모델 학습 함수
def train_random_forest_multilabel(records):
    df = pd.DataFrame([r.dict() for r in records])

    for r in records:
        print("sideEffects:", r.sideEffects)

    # 부작용 boolean 생성
    for effect in LABELS:
        df[effect] = df["sideEffects"].apply(lambda effects: effect in effects if effects else False)

    # mood 인코딩
    mood_map = {"나쁨": 0, "보통": 1, "좋음": 2}
    df["moodEncoded"] = df["mood"].map(mood_map)

    # feature & target
    X = df[["medicineId", "fatigueLevel", "dizzinessLevel", "moodEncoded", "sleepHours"]]
    Y = df[LABELS]

    combined = pd.concat([X, Y], axis=1).drop_duplicates()

    X = combined[["medicineId", "fatigueLevel", "dizzinessLevel", "moodEncoded", "sleepHours"]]
    Y = combined[LABELS]

    # 학습 가능한 라벨
    label_counts = {label: Y[label].sum() for label in LABELS}
    print("부작용 발생 수:", label_counts)

    learnable_labels = [label for label, count in label_counts.items() if count >= 2]
    if not learnable_labels:
        print("학습 가능한 라벨이 없습니다.")
        return "학습 불가: 양성 라벨 부족"

    # 학습용 Y
    Y = Y[learnable_labels]
    print("최종 학습 라벨:", Y.columns.tolist())

    # 모델 학습
    base_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model = MultiOutputClassifier(base_clf)
    model.fit(X, Y)

    # 저장
    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "labels": learnable_labels}, f)

    return "멀티라벨 부작용 예측 모델 학습 완료"

# 사용자별 학습 약 목록 로딩 


def predict_side_effect_proba(model, input_data, all_model_labels, user_labels):
    X = pd.DataFrame([input_data])

    try:
        probas = model.predict_proba(X)
        print(f"probas 타입: {type(probas)}, 길이: {len(probas)}")
        print(f"probas[0] 내용 예시: {probas[0]}")
    except Exception as e:
        print("model.predict_proba 호출 중 오류 발생:", e)
        raise

    result = {}
    for i, label in enumerate(all_model_labels):
        if label not in user_labels:
            continue  # 사용자가 이 약에 대해 학습한 라벨만 예측 결과에 포함

        try:
            proba_i = probas[i][0]
            if len(proba_i) == 2:
                result[label] = round(proba_i[1] * 100, 2)
            elif len(proba_i) == 1:
                pred_class = model.estimators_[i].classes_[0]
                result[label] = 100.0 if pred_class == 1 else 0.0
            else:
                result[label] = 0.0
        except Exception as e:
            print(f"[예측 오류] {label}: {e}")
            result[label] = 0.0

    return result

# 메인 예측 함수
def predict_side_effect_batch(request):
    try:
        data = request.data
        user_id = data[0].userId if data else None
        if not user_id:
            raise ValueError("예측 요청에 userId가 포함되어야 합니다.")

        with open(RF_MODEL_PATH, "rb") as f:
            model_bundle = pickle.load(f)
            model = model_bundle["model"]
            trained_model_labels = model_bundle["labels"]

        trained_ids = get_trained_medicine_ids_for_user(user_id)
        print(f"사용자 {user_id} 학습된 약 ID 목록: {trained_ids}")

        results = []

        for d in data:
            if d.medicineId not in trained_ids:
                print(f"{d.medicineName} 은(는) 사용자 {user_id}의 학습 대상 아님 -> 예측 생략")
                continue

            trained_labels = get_trained_labels_for_medicine(user_id, d.medicineId)
            print(f"약 {d.medicineName}에 대한 사용자 학습 라벨: {trained_labels}")

            input_data = {
                "medicineId": d.medicineId,
                "fatigueLevel": d.fatigueLevel,
                "dizzinessLevel": d.dizzinessLevel,
                "moodEncoded": d.moodEncoded,
                "sleepHours": d.sleepHours
            }

            proba = predict_side_effect_proba(
                model, input_data,
                all_model_labels=trained_model_labels,
                user_labels=trained_labels
            )

            messages = [f"{k}을(를) 유발할 확률이 {v}%" for k, v in proba.items() if v >= 10]
            if not messages:
                feedback = f"{d.medicineName}은(는) 현재 건강 상태 기준으로 특별한 부작용 가능성이 낮아요."
            else:
                feedback = f"{d.medicineName}은(는) " + ", ".join(messages) + " 가능성이 있어요."

            results.append(SideEffectPredictResponse(
                medicineId=d.medicineId,
                medicineName=d.medicineName,
                probabilities=proba,
                feedback=feedback
            ))

        if not results:
            return {"message": "해당 사용자가 학습한 약이 없어요."}

        return {"result": results}

    except Exception as e:
        print("전체 예측 처리 중 예외 발생:", e)
        raise

    
# LSTM - 피로도 트렌드 예측

def train_lstm(records):
    if len(records) < 4:
        raise ValueError("학습을 위해 최소 4개의 기록이 필요합니다.")
    
    X, y = [], []
    mood_map = {"나쁨": 0, "보통": 1, "좋음": 2}

    for i in range(len(records) - 3):
        seq = [[
            r["fatigueLevel"],
            r["dizzinessLevel"],
            mood_map.get(r["mood"], 1),
            r["sleepHours"]
        ] for r in records[i:i+3]]
        target = records[i + 3]["fatigueLevel"]
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    # 스케일링
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, 4)).reshape(X.shape)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # LSTM 모델 학습
    model = Sequential([
        LSTM(64, input_shape=(3, 4)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_scaled, y, epochs=50, verbose=0)
    model.save(LSTM_MODEL_PATH)

    return "LSTM 피로도 예측 모델 학습 완료"

def predict_fatigue(records):
    if len(records) < 3:
        raise ValueError("예측을 위해 최소 3개의 기록이 필요합니다.")
    
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)
    mood_map = {"나쁨": 0, "보통": 1, "좋음": 2}

    last_seq = np.array([[[
        r["fatigueLevel"],
        r["dizzinessLevel"],
        mood_map.get(r["mood"], 1),
        r["sleepHours"]
    ] for r in records[-3:]]])

    scaled_input = scaler.transform(last_seq.reshape(-1, 4)).reshape(1, 3, 4)
    prediction = model.predict(scaled_input, verbose=0)
    return float(np.round(prediction[0][0], 2))
