#!/bin/bash

set -e

SERVER_IP="129.211.167.133"
SERVER_USER="root"
SERVER_PATH="/root/TCM-RAG"

echo "===== 1. 进入项目目录 ====="
cd ~/TCM-RAG

echo "===== 2. 构建前端 ====="
cd tcm-rag-frontend
npm run build

echo "===== 3. 返回项目根目录 ====="
cd ..

echo "===== 4. 上传后端文件 ====="
scp api_server.py ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/
scp llm_client.py ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/

echo "===== 5. 上传前端 dist ====="
scp -r tcm-rag-frontend/dist ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/tcm-rag-frontend/

echo "===== 6. 重启服务器服务 ====="
ssh ${SERVER_USER}@${SERVER_IP} "systemctl restart tcm-rag && systemctl status tcm-rag --no-pager"

echo "===== 7. 部署完成 ====="
echo "请浏览器强制刷新后访问： http://wangzai1.top"