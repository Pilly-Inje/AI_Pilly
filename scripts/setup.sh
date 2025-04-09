#!/bin/bash

echo "[SETUP] FastAPI 가상환경 및 의존성 설치 시작"

APP_DIR=/home/ec2-user/fastapi-app
TMP_DIR=/home/ec2-user/tmp
export TMPDIR=$TMP_DIR
export PIP_CACHE_DIR=$TMP_DIR/pip-cache
export PIP_NO_CACHE_DIR=false 

sudo chown -R ec2-user:ec2-user $APP_DIR

mkdir -p "$APP_DIR/logs"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

cd $APP_DIR || {
  echo "[SETUP] 디렉토리 $APP_DIR 존재하지 않음"
  exit 1
}



if [ -d "venv" ]; then
  echo "[SETUP] 기존 venv 삭제"
  rm -rf venv
fi

python3 -m venv venv || {
  echo "[SETUP] venv 생성 실패"
  exit 1
}

source venv/bin/activate || {
  echo "[SETUP] 가상환경 활성화 실패"
  exit 1
}

if ! command -v pip &> /dev/null; then
  echo "[SETUP] pip 명령어 없음, 설치 시도"
  sudo yum install python3-pip -y || exit 1
fi

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

rm -rf $PIP_CACHE_DIR

echo "[SETUP] 패키지 설치 완료"
