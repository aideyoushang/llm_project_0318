from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from rag_service.agent.state import AgentState
from rag_service.modules.agent_tools import AGENT_TOOLS
from rag_service.modules.agent import get_llm

class RouteIntent(BaseModel):
    intent: Literal["direct_rag", "complex_reasoning", "chitchat"] = Field(
        ...,
        description="问题的意图分类。direct_rag: 简单的单点查询；complex_reasoning: 需要对比、条件约束或多跳推理的复杂问题；chitchat: 无关业务的闲聊。"
    )

def intent_router_node(state: AgentState) -> dict:
    """
    路由节点：判断用户的问题是需要走简单的 RAG 还是复杂的 Agent ReAct。
    """
    print("[Node: Router] 正在进行意图分类...")
    llm = get_llm()
    # 强制模型输出结构化的 JSON 来表示意图
    structured_llm = llm.with_structured_output(RouteIntent)
    
    # 提取最新的用户问题
    messages = state["messages"]
    latest_msg = messages[-1].content if messages else ""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个酒店预订系统的高级意图路由器。请分析用户问题，决定它应该走哪条处理链路：\n"
                   "- direct_rag: 简单的问题，只需一次检索即可回答（如：酒店有 wifi 吗？/ Riverside 酒店评分多少？）\n"
                   "- complex_reasoning: 复杂问题，需要多次检索、对比或条件组合（如：比较 A 酒店和 B 酒店的缺点 / 找一个既在市中心又适合带小孩的评价最高的酒店）\n"
                   "- chitchat: 日常打招呼（如：你好 / 谢谢）"),
        ("human", "{question}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"question": latest_msg})
    
    intent = result.intent if result else "direct_rag"
    print(f"[Node: Router] 分类结果: {intent}")
    return {"intent": intent}

def agent_reasoning_node(state: AgentState) -> dict:
    """
    ReAct 推理节点：绑定工具的大模型，负责决策和调用。
    """
    print("[Node: Agent] 正在进行 ReAct 推理决策...")
    llm = get_llm()
    # 给模型绑定工具，使其具备 Tool Calling 能力
    llm_with_tools = llm.bind_tools(AGENT_TOOLS)
    
    # 注入系统提示词
    sys_msg = SystemMessage(content=(
        "你是一个高级酒店分析 Agent。你需要通过调用工具来逐步收集信息，并回答复杂的对比或条件约束问题。\n"
        "如果你发现收集的信息不足，请继续调用工具；如果信息已经足够，请直接输出最终答案，不要再调用工具。"
    ))
    
    # 将系统消息放在开头，然后是当前的对话历史
    messages = [sys_msg] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    
    # 返回新的消息追加到状态中
    return {"messages": [response]}

def direct_rag_node(state: AgentState) -> dict:
    """
    单点 RAG 节点：对于简单问题，快速执行单次检索后回答，不走复杂的工具循环以节省 Token 和时间。
    """
    print("[Node: Direct RAG] 正在执行快速单步检索...")
    from rag_service.modules.agent_tools import search_reviews_by_semantic
    
    messages = state["messages"]
    latest_msg = messages[-1].content
    
    # 直接调用语义检索工具
    context = search_reviews_by_semantic.invoke({"query": latest_msg})
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个酒店客服。请根据以下参考信息，简洁准确地回答用户问题。\n\n参考信息：\n{context}"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": latest_msg})
    
    return {"messages": [response]}

def chitchat_node(state: AgentState) -> dict:
    """处理闲聊的节点"""
    print("[Node: Chitchat] 正在处理闲聊...")
    llm = get_llm()
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
