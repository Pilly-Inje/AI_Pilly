#!/bin/bash

echo "[DEPLOY] FastAPI 서버 배포 시작"

APP_DIR=/home/ec2-user/fastapi-app
cd $APP_DIR || {
  echo "[DEPLOY] 디렉토리 $APP_DIR 존재하지 않음"
  exit 1
}

mkdir -p "$APP_DIR/logs"

source venv/bin/activate || {
  echo "[DEPLOY] 가상환경 활성화 실패"
  exit 1
}

pkill -f gunicorn || true

nohup gunicorn -k uvicorn.workers.UvicornWorker main:app \
  --chdir app \
  --bind 0.0.0.0:8000 \
  > "$APP_DIR/logs/gunicorn.log" 2>&1 &

sleep 3
if pgrep -f gunicorn > /dev/null; then
  echo "[DEPLOY] Gunicorn 실행 성공"
else
  echo "[DEPLOY] Gunicorn 실행 실패. 로그 출력:"
  cat "$APP_DIR/logs/gunicorn.log"
  exit 1
fi
