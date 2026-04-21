import json

# ==========================================
# 1. 工具定义 (Tools)
# 这是 Agent 可以感知和调用的外部能力
# ==========================================
def search_hotels_by_location(location: str) -> str:
    """模拟根据地点搜索酒店的基础信息"""
    print(f"\n[Tool Execution] 正在搜索位于 '{location}' 的酒店...")
    # 模拟从数据库或 BM25 检索到的结果
    return json.dumps([
        {"hotel_id": "H001", "name": f"{location} Grand Hotel", "overall_rating": 4.5},
        {"hotel_id": "H002", "name": f"{location} Budget Inn", "overall_rating": 3.2},
        {"hotel_id": "H003", "name": f"{location} Riverside Suites", "overall_rating": 4.8}
    ], ensure_ascii=False)

def get_hotel_reviews(hotel_id: str) -> str:
    """模拟获取特定酒店的详细评论"""
    print(f"\n[Tool Execution] 正在获取酒店 '{hotel_id}' 的详细评论...")
    # 模拟根据 ID 从向量数据库或原始数据中获取详细评论
    if hotel_id == "H001":
        return "位置很好，但房间有点吵。性价比中等。"
    elif hotel_id == "H003":
        return "非常棒的体验，风景优美，服务一流，虽然价格偏高但物有所值。"
    return "没有找到评论。"

# 工具注册表，供模型调用
TOOLS = {
    "SearchHotelsByLocation": search_hotels_by_location,
    "GetHotelReviews": get_hotel_reviews
}

# ==========================================
# 2. ReAct Prompt 模板
# 严格约束模型的思考方式和输出格式
# ==========================================
REACT_PROMPT = """你是一个智能酒店顾问 Agent。你可以使用以下工具来回答用户的问题：

1. SearchHotelsByLocation(location: str): 根据地点查找当地酒店及其评分。
2. GetHotelReviews(hotel_id: str): 根据酒店ID获取该酒店的详细评论内容。

为了解决问题，你必须遵循以下严格的思考过程格式：

Question: 用户提出的问题
Thought: 我需要思考为了回答这个问题我应该采取什么行动。
Action: 要执行的工具名称（必须是 SearchHotelsByLocation 或 GetHotelReviews 之一）
Action Input: 传递给工具的参数（仅填参数值即可）
Observation: 工具执行后返回的结果（这个结果我会提供给你，你不用自己生成）
... (Thought/Action/Action Input/Observation 的循环可以重复多次)
Thought: 我现在已经收集到足够的信息来回答最终问题了。
Final Answer: 提供给用户的最终答案。

---

现在开始！

Question: {question}
"""

# ==========================================
# 3. 模拟大模型 (LLM Mock)
# 这里用预设的剧本模拟大模型的流式推理过程，
# 实际上你需要把 prompt 发给 Qwen2.5 并解析其输出
# ==========================================
def mock_llm_generate(prompt: str, history: list) -> str:
    """模拟 LLM 的生成过程，返回预设好的回答"""
    step = len(history) // 2 # 简单判断进行到哪一步了
    
    if step == 0:
        return "Thought: 用户问的是某个地方（比如这个地方）哪个酒店最好。我需要先找出这个地方有哪些酒店，并看看它们的评分。\nAction: SearchHotelsByLocation\nAction Input: 这个地方"
    elif step == 1:
        return "Thought: 根据评分，'Riverside Suites' (H003) 的评分最高（4.8分），其次是 'Grand Hotel' (H001，4.5分）。为了确定它为什么最好，我需要看看 H003 的具体评论。\nAction: GetHotelReviews\nAction Input: H003"
    elif step == 2:
        return "Thought: 评论显示 H003 确实很棒，服务和风景都很好，虽然有点贵。现在我已经有足够的信息了。\nFinal Answer: 根据检索，这个地方最好的酒店是 'Riverside Suites'（评分高达4.8分）。用户评价显示它能提供非常棒的体验，风景优美且服务一流。虽然价格可能偏高，但物有所值。另外，'Grand Hotel' 也是一个不错的选择（4.5分），但相比之下可能没有那么惊艳。"
    return "Thought: 我不知道该做什么了。\nFinal Answer: 抱歉，我无法回答这个问题。"

# ==========================================
# 4. Agent 执行循环 (The ReAct Loop)
# 负责解析模型输出、调用工具、将结果喂回给模型
# ==========================================
def run_react_agent(question: str):
    print(f"====== 开始处理问题: {question} ======")
    
    # 构造初始 prompt
    current_prompt = REACT_PROMPT.format(question=question)
    history = [] # 记录整个过程
    
    max_iterations = 5
    for i in range(max_iterations):
        print(f"\n--- 迭代第 {i+1} 轮 ---")
        
        # 1. LLM 生成 Thought 和 Action
        # 实际代码这里是：response = qwen_model.generate(current_prompt)
        llm_output = mock_llm_generate(current_prompt, history)
        print(f"LLM 输出:\n{llm_output}")
        
        # 2. 解析 LLM 输出
        if "Final Answer:" in llm_output:
            # 找到最终答案，结束循环
            final_answer = llm_output.split("Final Answer:")[1].strip()
            print(f"\n====== 最终结果 ======\n{final_answer}")
            break
            
        elif "Action:" in llm_output and "Action Input:" in llm_output:
            # 提取工具名和参数
            lines = llm_output.split('\n')
            action = next(line.split("Action:")[1].strip() for line in lines if "Action:" in line)
            action_input = next(line.split("Action Input:")[1].strip() for line in lines if "Action Input:" in line)
            
            # 3. 执行对应的工具
            if action in TOOLS:
                observation = TOOLS[action](action_input)
            else:
                observation = f"错误: 找不到工具 {action}"
                
            print(f"Observation: {observation}")
            
            # 4. 将过程记录到历史中，并更新 prompt
            history.append(llm_output)
            history.append(f"Observation: {observation}")
            
            # 将最新的观察结果拼接到 prompt 中，让 LLM 进行下一轮思考
            current_prompt += f"{llm_output}\nObservation: {observation}\n"
        else:
            print("模型输出格式错误，无法解析 Action。")
            break

if __name__ == "__main__":
    # 假设用户问了这样一个复杂问题
    run_react_agent("这个地方的酒店哪个最好？它们有什么优缺点？")
