#!/bin/bash

echo "[DEPLOY] FastAPI 서버 배포 시작"

cd /home/ec2-user/fastapi-app
source venv/bin/activate

# 기존 gunicorn 종료
pkill -f gunicorn || true

# FastAPI 실행
nohup gunicorn -k uvicorn.workers.UvicornWorker main:app \
  --chdir app \
  --bind 0.0.0.0:8000 \
  > /home/ec2-user/fastapi-app/logs/gunicorn.log 2>&1 &

# 실행 확인
sleep 3
if pgrep -f gunicorn > /dev/null; then
  echo "[DEPLOY] Gunicorn 실행 성공"
else
  echo "[DEPLOY] Gunicorn 실행 실패. 로그 출력:"
  cat /home/ec2-user/fastapi-app/logs/gunicorn.log
  exit 1
fi