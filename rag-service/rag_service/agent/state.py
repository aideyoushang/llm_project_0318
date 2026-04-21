from typing import Annotated, TypedDict, Sequence, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    LangGraph 的状态机定义，贯穿整个图的执行过程。
    - messages: 使用 add_messages 合并逻辑，保留多轮对话和工具调用历史
    - intent: 路由分类的结果 (direct_rag, complex_reasoning, chitchat)
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: Literal["direct_rag", "complex_reasoning", "chitchat", "unknown"]
