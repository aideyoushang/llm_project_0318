from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from rag_service.agent.state import AgentState
from rag_service.agent.nodes import (
    intent_router_node, 
    agent_reasoning_node, 
    direct_rag_node,
    chitchat_node
)
from rag_service.modules.agent_tools import AGENT_TOOLS

def should_continue(state: AgentState) -> str:
    """
    判断 Agent 是否需要继续调用工具。
    检查最新的一条消息（AIMessage）是否带有 tool_calls 属性。
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("[Workflow] 模型请求调用工具，进入 Tool Node")
        return "tools"
    
    print("[Workflow] 模型已输出最终答案，结束流程")
    return END

def route_by_intent(state: AgentState) -> str:
    """
    根据意图分类结果，将图的执行流向不同的节点。
    """
    intent = state.get("intent", "direct_rag")
    if intent == "complex_reasoning":
        return "agent_reasoning"
    elif intent == "chitchat":
        return "chitchat"
    else:
        # 默认为 direct_rag
        return "direct_rag"

def create_agent_workflow() -> StateGraph:
    """
    构建企业级 LangGraph 工作流
    """
    workflow = StateGraph(AgentState)
    
    # 1. 添加节点
    workflow.add_node("intent_router", intent_router_node)
    workflow.add_node("agent_reasoning", agent_reasoning_node)
    workflow.add_node("direct_rag", direct_rag_node)
    workflow.add_node("chitchat", chitchat_node)
    
    # ToolNode 是 LangGraph 内置的，专门用来执行我们定义的 AGENT_TOOLS
    tool_node = ToolNode(AGENT_TOOLS)
    workflow.add_node("tools", tool_node)
    
    # 2. 定义边和路由逻辑
    # 入口连接到路由节点
    workflow.add_edge(START, "intent_router")
    
    # 根据 router 的结果，动态分发到不同的处理节点
    workflow.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "agent_reasoning": "agent_reasoning",
            "direct_rag": "direct_rag",
            "chitchat": "chitchat"
        }
    )
    
    # direct_rag 和 chitchat 执行完直接结束
    workflow.add_edge("direct_rag", END)
    workflow.add_edge("chitchat", END)
    
    # ReAct 核心循环：Agent 节点执行完后，判断是调用工具还是结束
    workflow.add_conditional_edges(
        "agent_reasoning",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # 工具执行完毕后，强制回到 Agent 节点进行下一步推理 (Observation -> Thought)
    workflow.add_edge("tools", "agent_reasoning")
    
    return workflow.compile()

# 暴露一个全局可用的 compiled graph 实例
agent_graph = create_agent_workflow()
