from typing import Any, List, Dict
from pydantic import BaseModel, Field
from heapq import nlargest

try:
    from langchain_core.tools import tool
except ImportError:
    # Fallback/Mock for local syntax checking if langchain is not installed yet
    def tool(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

from rag_service.modules.retriever import RetrieverModule
from rag_service.modules.ranker import RankerModule

# 单例模式：全局共享一个 RetrieverModule 和 RankerModule 实例
_retriever_instance = None
_ranker_instance = None

def get_retriever() -> RetrieverModule:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RetrieverModule()
        _retriever_instance._ensure_loaded()
    return _retriever_instance

def get_ranker() -> RankerModule:
    global _ranker_instance
    if _ranker_instance is None:
        _ranker_instance = RankerModule()
    return _ranker_instance

class SearchHotelReviewsInput(BaseModel):
    query: str = Field(description="需要检索的具体关键词或自然语言句子")
    enable_bm25: bool = Field(default=True, description="是否开启 BM25 关键词检索。当查询包含具体实体（如 wifi, 培根, 酒店名）时设为 true。")
    enable_vector: bool = Field(default=True, description="是否开启向量语义检索。当查询是抽象语义（如 '适合带小孩', '感觉压抑'）时设为 true。")
    enable_summary: bool = Field(default=False, description="是否开启摘要检索。仅当需要了解酒店整体评分、总结性优缺点时设为 true。")

@tool("search_hotel_reviews", args_schema=SearchHotelReviewsInput)
def search_hotel_reviews(query: str, enable_bm25: bool = True, enable_vector: bool = True, enable_summary: bool = False) -> str:
    """
    【超级检索工具】当你需要查找酒店评论、设施信息或整体评价时，使用此工具。
    你可以通过参数控制使用关键词检索(BM25)、语义检索(Vector)或摘要检索(Summary)。如果不确定，建议同时开启 bm25 和 vector。
    返回的结果已经过底层融合与重排，并带有严格的 [doc_X] 编号，请在回答时务必引用。
    """
    retriever = get_retriever()
    
    rrf_k = 60
    fused: Dict[str, Dict[str, Any]] = {}

    def add_ranked(items: List[Any], route_weight: float) -> None:
        for item in items:
            score = route_weight / (rrf_k + item.rank)
            slot = fused.get(item.key)
            if slot is None:
                slot = {
                    "text": item.text,
                    "metadata": dict(item.metadata),
                    "score": 0.0,
                    "sources": {},
                }
                fused[item.key] = slot
            slot["score"] += score
            slot["sources"][item.source] = {"rank": item.rank, "w": route_weight}

    # 多路召回与融合 (设置 RRF 权重：Vector 稍高，BM25 基础，Summary 辅助)
    if enable_bm25:
        add_ranked(retriever._retrieve_bm25(query, top_k=20), route_weight=1.0)
    if enable_vector:
        add_ranked(retriever._retrieve_vector(query, top_k=20), route_weight=1.2)
    if enable_summary:
        add_ranked(retriever._retrieve_summary(query, top_k=10), route_weight=0.6)

    if not fused:
        return "未找到相关信息 (No relevant information found)。请尝试修改查询词或更换检索策略。"

    # 提取并按 score 排序
    top_candidates = nlargest(20, fused.items(), key=lambda kv: float(kv[1]["score"]))
    
    candidates_list = []
    for _, payload in top_candidates:
        candidates_list.append({
            "text": payload["text"],
            "metadata": payload["metadata"],
            "score": float(payload["score"]),
            "sources": payload.get("sources", {})
        })
        
    # 重排 (Rerank)
    ranker = get_ranker()
    ranked_candidates = ranker._heuristic_rerank(query, candidates_list, recency_level="none")
    
    # 截取最终 Top 8
    final_results = ranked_candidates[:8]
    
    # 格式化输出与严格引用埋点
    formatted = ["【系统提示：请严格使用 [doc_X] 引用以下证据回答用户。如果回答用到了某条证据，必须在句子末尾加上对应的 [doc_X]】"]
    
    for idx, c in enumerate(final_results, start=1):
        meta = c.get("metadata", {})
        hotel_id = meta.get("hotel_id", "Unknown")
        rating = meta.get("overall", "N/A")
        sources_str = ", ".join(c.get("sources", {}).keys())
        
        formatted.append(
            f"[doc_{idx}] (Hotel ID: {hotel_id} | 综合评分: {rating} | 召回来源: {sources_str})\n"
            f"原文: {c['text']}"
        )
        
    return "\n\n---\n\n".join(formatted)

# 暴露工具列表供 Agent 使用
AGENT_TOOLS = [
    search_hotel_reviews
]
