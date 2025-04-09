#!/bin/bash

echo "[DEPLOY] FastAPI 서버 배포 시작"

cd /home/ec2-user/fastapi-app || {
  echo "[DEPLOY] 디렉토리 /home/ec2-user/fastapi-app 존재하지 않음"
  exit 1
}

source venv/bin/activate || {
  echo "[DEPLOY] 가상환경 활성화 실패"
  exit 1
}

mkdir -p /home/ec2-user/fastapi-app/logs

pkill -f gunicorn || true

nohup gunicorn -k uvicorn.workers.UvicornWorker main:app \
  --chdir app \
  --bind 0.0.0.0:8000 \
  > /home/ec2-user/fastapi-app/logs/gunicorn.log 2>&1 &

sleep 3
if pgrep -f gunicorn > /dev/null; then
  echo "[DEPLOY] Gunicorn 실행 성공"
else
  echo "[DEPLOY] Gunicorn 실행 실패 로그 출력:"
  cat /home/ec2-user/fastapi-app/logs/gunicorn.log
  exit 1
fi