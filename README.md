# TCM-ARG

> A Large Model-Based Intelligent Question-Answering System for Traditional Chinese Medicine

TCM-ARG 是一个基于大语言模型（LLM）与检索增强生成（RAG）框架构建的中医药智能问答系统。  
系统面向中医药领域知识问答场景，结合结构化中医药数据、语义检索与大模型生成能力，实现对中药、方剂、证候、症状等内容的智能问答，并提供具有依据的回答结果。

---

## 1. Project Background

中医药知识体系具有专业性强、概念抽象、表达方式多样等特点。传统的信息检索方式通常依赖关键词匹配或人工查阅，难以高效满足自然语言问答需求。

随着大语言模型的发展，智能问答系统在通用领域取得了较好效果，但在中医药等垂直领域中仍存在一些典型问题：

- 缺乏领域知识支撑，容易产生“幻觉”
- 回答过程缺少证据支持，可信度不足
- 通用模型对中医药术语、证候语义理解有限

为解决上述问题，本项目引入 **RAG（Retrieval-Augmented Generation）** 框架，将中医药知识库与大语言模型结合，在生成答案前先进行知识检索，从而提升系统回答的准确性、专业性与可解释性。

---

## 2. Project Objective

本项目旨在设计并实现一个面向中医药领域的智能问答系统，使用户能够通过自然语言方式查询中医药相关知识，并获得清晰、准确、具有依据的回答。

系统主要目标包括：

- 支持用户对中药、方剂、证候、症状等内容进行自然语言提问
- 构建可供检索的中医药知识库
- 结合语义检索与大语言模型生成问答结果
- 在前端页面展示问题、答案及相关证据信息
- 支持本地运行与服务器部署

---

## 3. Data Sources

本系统使用的中医药数据主要来源于整理后的结构化表格数据，涵盖以下内容：

- **中药基础信息**
  - 药材名称
  - 性味
  - 归经
  - 功效
  - 主治
  - 药材分类

- **方剂信息**
  - 方剂名称
  - 方剂组成中药
  - 主治症状
  - 主治证候
  - 来源

- **中药-化合物关联数据**
  - 中药与单体成分对应关系

- **化合物-靶点关联数据**
  - 单体成分与基因/蛋白靶点关系

- **中医证候与中医症状数据**
  - 证候名称、定义
  - 症状名称、属性、部位等

- **中药转录组学相关数据**
  - 基因表达变化信息
  - 细胞系
  - logFC、P value 等字段

这些数据经过统一清洗、字段规范化与结构整理后，用于构建知识库与检索索引。

---

## 4. System Overview

系统整体采用“**前端交互 + 后端检索生成 + 知识库支撑**”的架构模式。  
核心流程如下：

1. 用户在前端页面输入自然语言问题
2. 前端将问题发送至后端接口
3. 后端对问题进行处理与向量化
4. 从中医药知识库中检索相关内容
5. 将检索结果作为上下文传入大语言模型
6. 大语言模型生成最终回答
7. 后端返回结果，前端展示答案及参考信息

系统通过将“检索”与“生成”结合，避免模型脱离知识库直接作答，从而提高回答质量。

---

## 5. System Architecture

系统主要由以下几个模块构成：

### 5.1 Data Processing Module

负责原始数据的清洗与预处理，包括：

- 字段筛选
- 缺失值处理
- 表头统一
- 文本规范化
- 不同数据表之间的结构整理

### 5.2 Knowledge Base Module

负责将整理后的中医药数据组织成可供查询与检索的知识库，包括：

- 中药知识数据
- 方剂知识数据
- 证候与症状数据
- 化合物与靶点关联数据
- 转录组学扩展数据

### 5.3 Retrieval Module

负责知识检索，包括：

- 文本切分与组织
- 文本向量化
- 向量索引构建
- 相似度检索
- Top-K 相关内容召回

### 5.4 LLM Generation Module

负责在检索结果基础上进行答案生成，包括：

- Prompt 组织
- 上下文拼接
- 大语言模型调用
- 问答结果输出

### 5.5 Frontend Interaction Module

负责用户交互界面，包括：

- 问题输入框
- 回答结果展示区
- 参考证据展示区
- 加载状态与基础交互反馈

### 5.6 Deployment Module

负责系统在服务器上的运行与发布，包括：

- 项目代码管理
- Python 环境配置
- 后端服务运行
- 前端静态资源部署
- Nginx 反向代理配置

---

## 6. Technology Stack

本系统采用前后端分离的实现思路，主要技术栈如下。

### 6.1 Backend

- **Python**  
  用于实现数据处理、知识检索、接口逻辑及模型调用

- **RAG (Retrieval-Augmented Generation)**  
  作为系统核心问答框架，将知识检索与答案生成结合

- **Embedding Model**  
  用于将问题与知识文本转换为向量表示，实现语义相似度检索

- **Vector Retrieval**  
  负责从知识库中召回与问题相关的内容

- **API Service**  
  用于提供前后端交互接口

### 6.2 Frontend

系统前端采用 Web 技术实现，用于构建用户可视化交互界面。  
根据当前实现路线，前端部分可描述为：

- **HTML**
- **CSS**
- **JavaScript**
- **Fetch / Ajax 请求**
- **前后端接口联调**

前端主要负责：

- 用户输入问题
- 调用后端接口
- 渲染问答结果
- 展示检索到的参考信息

### 6.3 Development & Deployment

- **Git**  
  用于项目版本管理

- **GitHub**  
  用于项目托管与代码同步

- **Linux Server**  
  用于后端服务部署

- **Python Virtual Environment**
  用于隔离项目运行环境

- **Nginx**
  用于静态资源托管与接口转发

- **Node.js（可选）**
  若后续前端涉及构建工具，可用于前端构建与打包

---

## 7. Project Structure

项目目录结构示例如下：


TCM-ARG/
│
├── data/                         # 原始数据与处理后的数据
│   ├── raw/                      # 原始表格数据
│   └── processed/                # 清洗后的数据
│
├── scripts/                      # 数据预处理脚本
│   ├── preprocess/
│   └── build_knowledge_base/
│
├── backend/                      # 后端核心代码
│   ├── api/                      # 接口层
│   ├── retriever/                # 检索模块
│   ├── llm/                      # 大模型调用模块
│   ├── service/                  # 业务逻辑
│   └── utils/                    # 工具函数
│
├── frontend/                     # 前端页面代码
│   ├── index.html
│   ├── css/
│   ├── js/
│   └── assets/
│
├── models/                       # 向量模型或模型配置
├── docs/                         # 项目说明文档、架构图等
├── app.py                        # 后端启动入口
├── requirements.txt              # Python 依赖
└── README.md                     # 项目说明文档


## 8. System Workflow

系统问答流程如下：

1. 用户在 Web 页面输入问题  
2. 前端通过 HTTP 请求调用后端接口  
3. 后端接收问题并进行预处理  
4. 使用向量模型对问题进行向量化  
5. 从知识库中检索相关内容  
6. 将检索结果作为上下文输入大语言模型  
7. 大语言模型生成最终回答  
8. 后端返回结果  
9. 前端页面展示问答结果  

---

## 9. Current Status

目前系统已完成基础功能实现：

- [x] 中医药数据整理与清洗  
- [x] 知识库构建  
- [x] 向量检索模块  
- [x] RAG 问答流程  
- [x] Web 前端界面  
- [x] 前后端接口联调  
- [ ] 检索效果优化  
- [ ] 系统性能优化  
- [ ] 服务器部署稳定性测试  

---

## 10. Local Development

### Clone Repository

```bash
git clone https://github.com/yourname/TCM-ARG.git
cd TCM-ARG
```

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Backend

```bash
python app.py
```

Default API address:

```
http://localhost:8000
```

### Run Frontend

Open directly:

```
frontend/index.html
```

Or run a simple server:

```bash
cd frontend
python -m http.server 8080
```

Access:

```
http://localhost:8080
```

---

## 11. Server Deployment

### 1. Login Server

```bash
ssh root@your_server_ip
```

### 2. Clone Project

```bash
cd /root
git clone https://github.com/yourname/TCM-ARG.git
cd TCM-ARG
```

### 3. Setup Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run Backend

```bash
python app.py
```

Run in background:

```bash
nohup python app.py > app.log 2>&1 &
```

Check logs:

```bash
tail -f app.log
```

### 5. Deploy Frontend

```bash
mkdir -p /var/www/tcm-arg
cp -r frontend/* /var/www/tcm-arg/
```

### 6. Configure Nginx

```nginx
server {
    listen 80;
    server_name your_domain_or_ip;

    location / {
        root /var/www/tcm-arg;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Restart Nginx:

```bash
nginx -t
systemctl restart nginx
```

---

## 12. Example Usage

### Example Question

```
人参有什么功效？
```

### Example Answer

```
人参具有大补元气、补脾益肺、生津养血、安神益智等功效。
```

### Example Evidence

```
药材名称：人参
性味：甘、微苦，微温
归经：脾、肺、心经
功效：大补元气，补脾益肺，生津，安神益智
```

---

## 13. Future Work

Future improvements may include:

- Adding more TCM literature data
- Optimizing vector retrieval strategies
- Supporting multi-turn dialogue
- Integrating knowledge graph enhanced QA
- Improving system response performance

---

## 14. Author

**Name:** Tian Yongwang  
**Major:** Data Science and Big Data Technology  
**Project:** Undergraduate Graduation Project  

---

## 15. License

This project is intended for **academic and educational purposes only**.
