import os
import sys

# 确保能导入 rag_service
sys.path.append(os.path.join(os.path.dirname(__file__), "rag-service"))

from langchain_core.messages import HumanMessage
from rag_service.agent.workflow import agent_graph

def run_test(query: str):
    print(f"\n==================================================")
    print(f"用户提问: {query}")
    print(f"==================================================")
    
    # 初始化状态
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # 运行 LangGraph，stream 模式可以让我们看到每一步的执行
    for output in agent_graph.stream(initial_state, stream_mode="updates"):
        # output 是一个字典，key 是刚刚执行完的节点名称
        for node_name, state_update in output.items():
            print(f"\n---> [流转日志] 节点 '{node_name}' 执行完毕")
            
            # 如果是意图路由节点，打印出意图
            if "intent" in state_update:
                print(f"     判定意图: {state_update['intent']}")
                
            # 如果有新消息产生（通常是 LLM 的回复或 Tool 的结果）
            if "messages" in state_update:
                latest_msg = state_update["messages"][-1]
                
                # 如果是模型调用了工具
                if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                    print(f"     决定调用工具: {latest_msg.tool_calls}")
                    
                # 如果是工具执行结果 (ToolMessage)
                elif latest_msg.type == "tool":
                    # 只打印前100个字符避免刷屏
                    print(f"     工具返回结果: {latest_msg.content[:100]}...")
                    
                # 如果是最终回复
                elif latest_msg.content and not getattr(latest_msg, "tool_calls", None):
                    print(f"     节点输出文本: {latest_msg.content[:200]}...")

if __name__ == "__main__":
    # 测试用例 1: 闲聊 (预期触发 chitchat_node)
    run_test("你好呀，你是谁？")
    
    # 测试用例 2: 简单事实查询 (预期触发 direct_rag_node)
    run_test("酒店提供免费的 wifi 吗？")
    
    # 测试用例 3: 复杂的对比与推理 (预期触发 intent_router -> agent_reasoning -> tools 循环)
    run_test("我想找一家在市中心的酒店，并且要求它的卫生评分（cleanliness）很高，你能帮我对比一下排名前两名的优缺点吗？")
