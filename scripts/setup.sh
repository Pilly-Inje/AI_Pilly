#!/bin/bash

echo "[SETUP] FastAPI 가상환경 및 의존성 설치 시작"

cd /home/ec2-user/fastapi-app

# 기존 가상환경 삭제 (선택)
if [ -d "venv" ]; then
  echo "[SETUP] 기존 venv 삭제"
  rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "[SETUP] requirements.txt 없음"
  exit 1
fi

echo "[SETUP] 패키지 설치 완료"