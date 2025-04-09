#!/bin/bash

echo "[SETUP] FastAPI 가상환경 및 의존성 설치 시작"

cd /home/ec2-user/fastapi-app || {
  echo "[SETUP] 디렉토리 /home/ec2-user/fastapi-app 존재하지 않음"
  exit 1
}

mkdir -p /home/ec2-user/fastapi-app/logs

if [ -d "venv" ]; then
  echo "[SETUP] 기존 venv 삭제"
  rm -rf venv
fi

python3 -m venv venv || {
  echo "[SETUP] 가상환경 생성 실패"
  exit 1
}

source venv/bin/activate || {
  echo "[SETUP] 가상환경 활성화 실패"
  exit 1
}

pip install --upgrade pip

if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt || {
    echo "[SETUP] requirements.txt 설치 실패"
    exit 1
  }
else
  echo "[SETUP] requirements.txt 없음"
  exit 1
fi

echo "[SETUP] 패키지 설치 완료"
