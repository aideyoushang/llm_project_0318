from typing import List, Dict, Any
import json
import traceback

# 假设的现代 LLM 接口，支持原生 Tool Calling 和多轮历史
class ModernLLM:
    def __init__(self):
        # 预设一个复杂的生产级剧本，包含格式错误和自愈
        self.call_count = 0
        
    def bind_tools(self, tools: List[Dict]):
        # 生产环境中，这里会将 Python 函数的签名转换为 JSON Schema 传给大模型底层
        pass
        
    def invoke(self, messages: List[Dict]) -> Dict:
        """模拟支持原生 Tool Calling 的大模型返回结果"""
        self.call_count += 1
        
        # 第一次：模型尝试调用工具，但模拟它犯了个错（比如参数名拼错了）
        if self.call_count == 1:
            return {
                "content": "", # 生产环境中，Tool Call 时 content 通常为空或只包含 Thought
                "tool_calls": [{
                    "id": "call_123",
                    "name": "SearchHotels",
                    "args": {"loc": "北京"} # 错误：我们的工具要求参数叫 'location'
                }]
            }
            
        # 第二次：模型收到了错误信息，自我纠正了参数名
        elif self.call_count == 2:
            return {
                "content": "刚才参数传错了，我现在修正一下。",
                "tool_calls": [{
                    "id": "call_456",
                    "name": "SearchHotels",
                    "args": {"location": "北京"} # 正确的参数
                }]
            }
            
        # 第三次：模型收到了正确结果，给出最终答案
        elif self.call_count == 3:
            return {
                "content": "根据检索，北京评分最高的酒店是...",
                "tool_calls": [] # 空代表无需再调用工具，任务结束
            }

# ==========================================
# 1. 生产级工具定义
# ==========================================
def search_hotels(location: str) -> str:
    """根据地点搜索酒店的基础信息"""
    if location == "北京":
        return json.dumps([{"hotel": "北京饭店", "score": 4.8}], ensure_ascii=False)
    return "没有找到"

# 工具的 JSON Schema 描述（生产中往往通过 Pydantic 自动生成）
TOOLS_SCHEMA = [
    {
        "name": "SearchHotels",
        "description": "根据地点查找当地酒店",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市名称"}
            },
            "required": ["location"]
        },
        "function": search_hotels
    }
]

# ==========================================
# 2. 生产级 Agent 执行器 (包含容错和上下文)
# ==========================================
def run_production_agent(user_input: str, chat_history: List[Dict]):
    print(f"====== 生产级 Agent 开始执行 ======")
    llm = ModernLLM()
    llm.bind_tools(TOOLS_SCHEMA)
    
    # 构造消息历史：System Prompt + 过往聊天记录 + 当前问题
    messages = [
        {"role": "system", "content": "你是一个智能酒店助理，请使用工具回答问题。"}
    ]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_input})
    
    max_steps = 5 # 防死循环兜底
    
    for step in range(max_steps):
        print(f"\n--- 第 {step+1} 步 ---")
        
        # 1. 调用大模型
        response = llm.invoke(messages)
        messages.append({"role": "assistant", "content": response["content"], "tool_calls": response.get("tool_calls", [])})
        
        # 2. 判断是否完成
        tool_calls = response.get("tool_calls", [])
        if not tool_calls:
            print(f"\n[Agent 结论]: {response['content']}")
            return response['content']
            
        # 3. 处理工具调用（包含异常捕获与自愈）
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"[LLM 决定调用工具]: {tool_name}, 参数: {tool_args}")
            
            tool_result = ""
            # 找到对应的工具
            tool_def = next((t for t in TOOLS_SCHEMA if t["name"] == tool_name), None)
            
            if not tool_def:
                # 容错：大模型瞎编了一个不存在的工具
                tool_result = f"Error: 找不到名为 {tool_name} 的工具，请检查工具列表。"
            else:
                try:
                    # 尝试执行工具
                    # 模拟参数验证（大模型传了 'loc' 而不是 'location'）
                    if "location" not in tool_args:
                        raise ValueError("缺少必填参数 'location'")
                        
                    func = tool_def["function"]
                    tool_result = func(**tool_args)
                    print(f"[工具执行成功]: {tool_result}")
                    
                except Exception as e:
                    # 容错核心：捕获异常，并把错误信息原封不动扔回给大模型
                    error_msg = str(e)
                    print(f"[工具执行报错 (将被大模型感知)]: {error_msg}")
                    tool_result = f"Tool Execution Error: {error_msg}. Please correct your arguments and try again."
            
            # 将工具执行结果（或报错信息）追加到消息流中
            messages.append({
                "role": "tool", 
                "tool_call_id": tool_call["id"],
                "content": tool_result
            })
            
    # 如果超出了最大步数，触发兜底
    print("\n[兜底机制触发]: 超过最大思考步数，强制返回。")
    return "抱歉，这个问题有点复杂，我暂时无法得出结论。"

if __name__ == "__main__":
    # 假设这是多轮对话
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！我是酒店顾问。"}
    ]
    run_production_agent("北京哪个酒店好？", history)