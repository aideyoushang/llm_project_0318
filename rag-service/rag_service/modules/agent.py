from typing import Any, AsyncGenerator, Dict, List, Optional
import json

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
except ImportError:
    pass

from rag_service.modules.agent_tools import AGENT_TOOLS
from rag_service.modules.runtime_config import load_runtime_config

# 全局共享的 Agent Executor 和 Memory
_agent_executor = None
_memory = None

def get_llm() -> "ChatOpenAI":
    rc = load_runtime_config()
    # Volcengine Ark 兼容 OpenAI 接口格式
    base_url = "https://ark.cn-beijing.volces.com/api/v3" 
    api_key = (rc.ark_api_key or "").strip()
    model = (rc.ark_model or "").strip()
    
    if not api_key or not model:
        raise ValueError("缺少 ARK_API_KEY 或 ARK_MODEL 环境变量配置")
        
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        max_retries=2
    )

def get_agent_executor() -> "AgentExecutor":
    global _agent_executor, _memory
    if _agent_executor is not None:
        return _agent_executor
        
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的酒店评价分析助手，名为 Sway_Agent。你的目标是基于给定的酒店评论和摘要，客观、准确地回答用户的问题。\n\n"
                   "【你的工作流程】\n"
                   "1. 分析用户问题，思考需要哪些信息。\n"
                   "2. 使用 `search_hotel_reviews` 工具查询相关信息。通过工具参数自主决定是否开启关键词(BM25)、语义(Vector)或摘要(Summary)检索。\n"
                   "3. 综合工具返回的带编号证据（格式如 [doc_1], [doc_2]），给出清晰的回答。\n\n"
                   "【严格引用规则】\n"
                   "- 回答时，必须且只能基于工具返回的 [doc_X] 证据。\n"
                   "- 在回答的每句话末尾，必须使用 [doc_X] 的形式标出你的信息来源。例如：这家酒店的床很舒服[doc_1]，但早餐选择较少[doc_2]。\n"
                   "- 如果查询结果不足以回答，诚实地告诉用户不知道。\n"
                   "- 绝不能在没有依据的情况下捏造评论事实。"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, AGENT_TOOLS, prompt)
    
    _memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    
    _agent_executor = AgentExecutor(
        agent=agent, 
        tools=AGENT_TOOLS, 
        memory=_memory,
        verbose=True,
        max_iterations=5
    )
    return _agent_executor

def run_agent(query: str) -> str:
    """
    运行 Agent，返回最终答案的字符串。
    （注：此方法是同步且非流式的，主要用于测试或简单接口）
    """
    executor = get_agent_executor()
    result = executor.invoke({"input": query})
    return result["output"]

# ----------------------------------------------------------------------
# 针对你现有的 SSE 流式接口，我们需要拦截 Agent 的内部事件并转化为 SSE
# ----------------------------------------------------------------------
async def stream_agent(query: str) -> AsyncGenerator[str, None]:
    """
    使用 LangChain 的 astream_events 流式运行 Agent，将中间思考、工具调用、以及最终回复
    封装为你现有的 SSE 事件格式（如：stage, answer_chunk, [DONE]）。
    """
    executor = get_agent_executor()
    
    yield f"data: {json.dumps({'type': 'stage', 'stage': 'agent_thinking', 'content': 'Agent 正在思考并规划检索...'})}\n\n"
    
    # 注意：这里需要环境安装了 langchain-core >= 0.1.14
    # 使用 astream_events 来捕获各种细粒度的事件
    events = executor.astream_events(
        {"input": query}, 
        version="v1"
    )
    
    async for event in events:
        kind = event["event"]
        
        # 1. 触发工具调用
        if kind == "on_tool_start":
            tool_name = event["name"]
            tool_input = event["data"].get("input", {})
            msg = f"正在调用工具: {tool_name}，参数: {tool_input}"
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'tool_call', 'content': msg})}\n\n"
            
        # 2. 工具调用完成
        elif kind == "on_tool_end":
            tool_name = event["name"]
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'tool_end', 'content': f'{tool_name} 调用完成并返回了结果'})}\n\n"
            
        # 3. LLM 开始流式输出最终回答
        elif kind == "on_chat_model_stream":
            # 我们只需要最终输出的 Token，不输出工具调用的 Token
            # 通过 run_id 或 tag 过滤，通常最终的输出属于最外层的 chain 触发的 chat model
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                yield f"data: {json.dumps({'type': 'answer_chunk', 'content': chunk.content})}\n\n"
                
    yield "data: [DONE]\n\n"
