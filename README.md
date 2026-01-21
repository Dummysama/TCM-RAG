# TCM-RAG

> A Large Model-Based Intelligent Question Answering System for Traditional Chinese Medicine

TCM-RAG 是一个**基于大语言模型（LLM）与检索增强生成（Retrieval-Augmented Generation, RAG）框架**的中医药智能问答系统。  
系统面向中医药领域，通过整合结构化中医药数据与大模型生成能力，实现对中药、方剂、证候、症状等知识的智能问答，并提供基于证据的回答结果。

---

## 1. Project Background

中医药知识体系复杂、专业性强，传统查询方式依赖人工检索，效率较低。  
随着大语言模型的发展，智能问答在通用领域已取得良好效果，但在垂直专业领域仍面临以下问题：

- 缺乏权威、结构化知识支撑，容易产生“幻觉”
- 难以对回答结果给出明确依据
- 通用模型对中医药专业语义理解能力有限

本项目引入 **RAG（检索增强生成）框架**，将中医药知识库与大语言模型相结合，提升问答的准确性与可信度。

---

## 2. System Overview

系统整体采用 **“检索 + 生成”** 的智能问答架构，主要流程如下：

1. 用户输入自然语言问题  
2. 系统对问题进行向量化表示  
3. 从中医药知识库中检索相关内容  
4. 将检索结果作为上下文输入大语言模型  
5. 生成基于证据的问答结果并返回给用户  

---

## 3. Data Sources

本系统使用的中医药数据主要包括：

- 中药基础信息（性味、归经、功效等）
- 方剂及其组成中药
- 中药—化合物—靶点关联数据
- 中医证候与中医症状数据
- 部分中药转录组学相关数据

数据经过清洗、规范化与结构整理，用于构建知识库与向量索引。

---

## 4. System Architecture

系统主要由以下模块组成：

- **Data Processing Module**
  - 数据清洗与格式统一
  - 文本字段规范化处理

- **Knowledge Base & Retrieval Module**
  - 文本向量化
  - 向量索引构建
  - 基于语义相似度的知识检索

- **Large Language Model Module**
  - 调用大语言模型接口
  - 融合检索结果进行答案生成

- **User Interaction Module**
  - 提供用户问题输入
  - 展示问答结果及参考信息

---

## 5. Implementation Workflow

项目整体实现流程如下：

1. 数据收集与预处理  
2. 构建中医药知识库  
3. 文本向量化与索引构建  
4. 实现基于相似度的检索功能  
5. 接入大语言模型进行回答生成  
6. 完成问答系统整体联调与测试  

---

## 6. Technology Stack

- **Programming Language**: Python  
- **Large Language Model**: API-based LLM  
- **Retrieval Framework**: RAG (Retrieval-Augmented Generation)  
- **Vector Storage**: 向量数据库 / 本地向量索引  
- **Frontend**: Web-based interface (optional)  

---

## 7. Project Structure


TCM-RAG/
│
├── data/                 # 原始数据与处理后的数据
├── scripts/              # 数据清洗与预处理脚本
├── retriever/            # 向量化与检索相关代码
├── llm/                  # 大语言模型调用模块
├── app/                  # 系统主程序
├── README.md             # 项目说明文档
zzz

## 8. Current Status

目前系统已完成基础功能实现，整体流程可正常运行，具体进展如下：

- [x] 中医药原始数据整理与结构化处理  
- [x] 核心字段清洗与规范化  
- [x] 中医药知识库构建  
- [x] 文本向量化与相似度检索实现  
- [x] 基于 RAG 框架的智能问答流程搭建  
- [ ] 检索与生成参数进一步优化  
- [ ] 前端交互界面完善（可选）  

---

## 9. Future Work

在当前实现基础上，系统仍具有进一步拓展与优化空间，主要包括：

- 引入更多中医药文献与非结构化文本数据  
- 优化向量检索策略，提高知识召回质量  
- 增强回答结果的可解释性与证据展示能力  
- 支持多轮对话与上下文理解  
- 对系统整体性能与用户体验进行优化  

---

## 10. Author

- **Name**: 田用旺  
- **Major**: Data Science and Big Data Technology  
- **Project Type**: Undergraduate Graduation Project  

---

## 11. License

This project is intended for **academic research and educational purposes only**.  
Commercial use is not permitted without authorization.
