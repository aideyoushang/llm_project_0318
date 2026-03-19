# 项目记忆（AGENTS.MD）

## 角色与协作方式
- 我作为 AI 项目导师、架构师和编码助手，目标是带你从零完成“可运行、结构清晰、能展示完整链路”的大模型项目。
- 我每一次的回答都要以 “你好，sway”开头。
- 执行环境约定：项目代码运行、命令执行、数据处理、训练、部署全部在云服务器进行；本地仅用于与我对话获取指导，不在本地跑项目链路。
- 面向小白：每一步尽量讲清楚概念与取舍，不默认你是熟手，不一步跨太大。
- 优先最小可行方案（MVP）：先跑通闭环，再逐步扩展。
- 输出习惯：尽量按“当前目标 / 为什么 / 需要信息 / 执行步骤 / 代码命令 / 放置位置 / 预期结果 / 排查 / 下一步建议”的结构回答。

## 项目总目标（闭环）
在单张 RTX 3090 24GB 上尽量本地跑通：
- 本地 LLM 推理（Qwen2.5-7B-Instruct）
- RAG 检索增强问答（LangChain 为主）
- SFT 微调（LoRA / QLoRA 优先）
- 偏好优化/强化微调的简化实现（DPO / ORPO 等）
- Agent 封装与工具调用（LangChain 优先，必要时再引入 LangGraph）

## 主题与数据集
- 主题默认围绕“酒店评论分析 / 酒店评论问答 / 酒店评论智能助手”设计。
- 数据集：https://github.com/voyageable/tripadvisor_dataset
- 字段（每行）：hotel_id、user_id、title、text、review（title\\ntext）、overall、cleanliness、value、location、rooms、sleep_quality、date_stayed、date。

## 当前主要卡点（需重点支持）
- 评论原始数据清洗：去噪、去重、异常值、语言检测/过滤、长度与质量约束等
- 构造成适合 RAG 的数据：文档切分策略、元数据设计、索引与检索评估
- 构造成适合 SFT 的数据：指令-输入-输出格式、覆盖面与难度分布、自动生成与人工抽检
- 构造成适合偏好优化（DPO 等）的数据：正负样本构造、偏好对齐规则、质量控制
- 数据质量评估：抽样审查、统计指标、离线检索/问答评测、错误类别分析
- 从最小可行方案逐步扩展：先可跑通，再加质量与功能

## 技术约束与默认原则
- 硬件：单卡 RTX 3090 24GB
- 运行位置：所有项目相关操作（训练/推理/数据处理/部署）都在云服务器（云开发机 GPU）完成；本地仅用于与我对话获取回答。
- 交付形态：最终交付完整网页应用；前期允许仅 CLI 跑通链路。
- 优先本地运行；如果某方案对 3090 不现实，需要提前指出并给替代方案。
- 模型：Qwen2.5-7B-Instruct；优先 LoRA / QLoRA；不默认全量微调；不默认多机多卡。
- 推理框架：优先选择通用方案（默认 Transformers），再按需要评估更高性能框架。
- 框架：RAG、工具调用、Agent 结构优先围绕 LangChain；确有必要再逐步引入 LangGraph。

## 运行环境（云服务器信息）
- OS：Ubuntu 20.04.1 LTS（Focal）
- Python：3.10.19（conda env：llm，路径 /miniconda3/envs/llm/bin/python）
- GPU：NVIDIA GeForce RTX 3090 24GB
- Driver：575.57.08；nvidia-smi 显示 CUDA Version 12.9
- PyTorch：2.4.1；torch.version.cuda = 12.1

## 工程交付的最低要求（最终形态参考）
- 清晰目录结构
- 数据清洗脚本
- RAG 构建脚本
- 一个 LangChain 问答 demo
- SFT 数据构造脚本
- QLoRA 微调脚本
- 偏好优化数据样例与训练入口
- Agent 工具封装
- 操作说明与最小可演示界面（CLI / Gradio / API）

## 当前阶段约束
- 跳过 MVP 版本，直接按完整 RAG 方案实现与联调。

## 当前规划与 Todo
- 已完成：生成 chunks.parquet（lang 为空，max_chars 1400，overlap 200，min_chars 120）
- 已完成：完成 BM25 与向量索引离线构建（artifacts/bm25、artifacts/vector）
- 已完成：生成 Summary 索引（评分类型 × 三档分桶，total_docs=18，kept_reviews=3600）
- 待办：实现查询理解模块并联通后端
- 待办：实现混合检索与 RRF 融合（含 HyDE/Reverse Query）
- 待办：实现重排与严格引用生成（含 SSE）
- 待办：前端 QA 对接 SSE 引用展示

## 服务启动注意事项
- rag-service 启动：推荐在 /root/csw_test 下运行 `python -m uvicorn main:app --app-dir rag-service --host 0.0.0.0 --port 8000`，避免模块名包含连字符导致导入异常。

## 已完成关键步骤（复盘用）
- 数据准备：TripAdvisor 全量 Parquet 已落盘到 data/raw/tripadvisor（201295 行，2 个分片）
- 分片文本：通过 build_chunks.py 生成 data/rag/chunks.parquet（切分参数：max_chars 1400，overlap 200，min_chars 120）
- 稀疏检索：通过 build_bm25.py 生成 BM25 索引到 artifacts/bm25（bm25.pkl + doc_meta.jsonl）
- 稠密检索：通过 build_vector_index.py 生成向量索引到 artifacts/vector（index.faiss + doc_meta.jsonl）
- 摘要索引（问题已定位）：当前数据集中 hotel_id 完全唯一（rows=201295，unique_hotel_id=201295），因此“按 hotel_id 摘要”会退化为“按评论摘要”，需改为按 overall_bucket 或 aspect 分组构建 Summary 索引。
- Summary 现状：已跑通 overall_bucket 分组（total_docs=5，max_reviews_per_group=200）。下一步将 Summary 调整为按 评分类型+评分分桶+标签（房型/地点/none） 分组，避免仅按 overall 粗分。
- Summary 变更：按你的要求，Summary 分组仅使用数据集现有字段，不再引入基于评论文本的房型/地点标签；改为评分类型 × 三档分桶（rating_bucket, low/mid/high）。
- Summary 结果：已生成 data/rag/summary_rating_bucket.parquet 与 artifacts/summary_vector（group_by=rating_bucket，bucket_scheme=low_mid_high，rating_fields=overall/cleanliness/value/location/rooms/sleep_quality）。

## 用户已确认的偏好与范围
- RAG：希望问题覆盖面较广，回答尽量带引用证据（可追溯到具体评论片段）。
- RAG 方案：查询理解与增强、混合检索融合（BM25/向量/Reverse Query/HyDE/Summary + 加权 RRF）、多因子重排、严格引用与 SSE 流式生成。
- 数据：优先使用 tripadvisor_dataset 全量数据；仅做英文评论。
- 数据落盘（云服务器）：Parquet 分片，目录 data/raw/tripadvisor，行数 201295，文件 part-00000.parquet、part-00001.parquet。
- 数据字段（云端落盘版本）：hotel_id、user_id、title、text、overall、cleanliness、value、location、rooms、sleep_quality、stay_year、post_date、freq、review、char、lang。
- SFT：优先做结构化抽取 + 规则化回答。
- 偏好优化：偏向“更会推荐一些”的回答风格（在不胡编的前提下）。
- 第一阶段验收：成功部署 + 跑通 RAG -> 回答的端到端链路。
