
# 基于大模型的中医药智能问答系统（TCM-ARG）
> Large Model-Based Intelligent Question-Answering System for Traditional Chinese Medicine

TCM-ARG 是一个基于大语言模型（LLM）与检索增强生成（Retrieval-Augmented Generation, RAG）框架构建的中医药智能问答系统。  
系统面向中医药知识问答场景，通过结合结构化中医药数据、语义检索以及大模型生成能力，实现对中药、方剂、证候、症状等内容的智能问答。

---

## 1. Project Background

中医药知识体系具有专业性强、表达方式复杂等特点，普通用户难以快速获取准确解释。  
传统检索方式主要依赖关键词搜索，难以支持自然语言问答。

随着大语言模型的发展，智能问答系统能够理解自然语言问题，但在垂直领域仍存在以下问题：

- 缺乏专业知识支撑，容易产生“幻觉”
- 回答结果缺少证据支持
- 通用模型对中医药术语理解能力有限

本项目引入 **RAG（Retrieval-Augmented Generation）** 框架，在生成答案前先检索相关知识，从而提高回答的准确性与可信度。

---

## 2. Project Objective

本项目旨在构建一个面向中医药领域的智能问答系统，使用户能够通过自然语言方式查询中医药相关知识，并获得清晰、准确的回答。

系统目标包括：

- 支持用户查询中药、方剂、证候和症状等信息
- 构建中医药知识库
- 实现语义检索与大模型结合的问答系统
- 提供 Web 前端界面进行交互
- 支持本地运行与服务器部署

---

## 3. Data Sources

系统使用的中医药数据主要包括：

- 中药基础信息（名称、性味、归经、功效等）
- 方剂数据（方剂组成、主治症状、证候等）
- 中药—化合物关联数据
- 化合物—靶点关联数据
- 中医证候与中医症状数据
- 中药转录组学数据

这些数据经过清洗与结构化处理后，用于构建系统知识库。

---

## 4. System Overview

系统整体流程如下：

1. 用户输入自然语言问题  
2. 前端将问题发送给后端接口  
3. 后端对问题进行向量化  
4. 从知识库中检索相关内容  
5. 将检索结果作为上下文输入大语言模型  
6. 大语言模型生成回答  
7. 返回结果并在前端展示  

---

## 5. System Architecture

系统主要包含以下模块：

### Data Processing Module
负责数据清洗与格式统一。

### Knowledge Base Module
负责构建中医药知识库。

### Retrieval Module
负责向量检索与相关内容召回。

### LLM Generation Module
调用大语言模型生成回答。

### Frontend Module
提供用户交互界面。

### Deployment Module
负责系统在服务器上的部署与运行。

---

## 6. Technology Stack

### Backend

- Python
- RAG (Retrieval-Augmented Generation)
- Embedding Model
- Vector Retrieval
- API Service

### Frontend

- HTML
- CSS
- JavaScript
- Fetch / Ajax 请求

### Deployment

- Linux Server
- Git / GitHub
- Python Virtual Environment
- Nginx

---

## 7. Project Structure

```
TCM-ARG/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── scripts/
│   ├── preprocess/
│   └── build_knowledge_base/
│
├── backend/
│   ├── api/
│   ├── retriever/
│   ├── llm/
│   ├── service/
│   └── utils/
│
├── frontend/
│   ├── index.html
│   ├── css/
│   ├── js/
│   └── assets/
│
├── models/
├── docs/
├── app.py
├── requirements.txt
└── README.md
```

---

## 8. System Workflow

系统问答流程：

1. 用户输入问题  
2. 前端调用后端 API  
3. 后端对问题进行向量化  
4. 在知识库中检索相关内容  
5. 将检索结果与问题一起输入模型  
6. 模型生成回答  
7. 返回并展示结果  

---

## 9. Current Status

当前系统完成情况：

- [x] 数据整理与清洗
- [x] 知识库构建
- [x] 向量检索模块
- [x] RAG问答流程
- [x] Web前端页面
- [ ] 系统性能优化
- [ ] 检索效果优化
- [ ] 服务器部署优化

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

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Backend

```bash
python app.py
```

### Run Frontend

```
frontend/index.html
```

---

## 11. Server Deployment

### Login Server

```bash
ssh root@your_server_ip
```

### Clone Project

```bash
git clone https://github.com/yourname/TCM-ARG.git
cd TCM-ARG
```

### Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Backend

```bash
nohup python app.py > app.log 2>&1 &
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

---

## 13. Future Work

未来可以进一步扩展：

- 引入更多中医药文献数据
- 优化检索算法
- 支持多轮对话
- 引入知识图谱增强问答
- 提升系统响应速度

---

## 14. Author

Name: Tian Yongwang  
Major: Data Science and Big Data Technology  

---

## 15. License

This project is intended for academic and educational purposes only.
