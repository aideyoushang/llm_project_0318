# llm

## 1) 部署本地推理服务（Transformers）

```bash
conda activate llm
pip install -U "transformers>=4.45" accelerate safetensors fastapi uvicorn
pip install -U bitsandbytes peft trl datasets
```

下载模型（示例）：

```bash
mkdir -p models
cd models
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir Qwen2.5-7B-Instruct --local-dir-use-symlinks False
```

启动：

```bash
cd /root/csw_test
python llm/serve_transformers.py --model /root/csw_test/models/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 9000
```

测试：

```bash
curl -s http://127.0.0.1:9000/healthz
curl -s -X POST http://127.0.0.1:9000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}' | python -m json.tool
```

## 2) 用 eval 结果构造 SFT 数据

```bash
cd /root/csw_test
python rag-service/eval_rag.py
python llm/sft_from_eval.py --in artifacts/eval/run_YYYYmmdd_HHMMSS.jsonl --out artifacts/sft/sft.jsonl
```

## 3) QLoRA SFT 训练

```bash
cd /root/csw_test
python llm/train_sft_qlora.py \
  --model /root/csw_test/models/Qwen2.5-7B-Instruct \
  --data artifacts/sft/sft.jsonl \
  --out artifacts/sft/ckpt
```

## 4) 用 eval 结果构造偏好数据（DPO）

```bash
cd /root/csw_test
python llm/pref_from_eval.py --in artifacts/eval/run_YYYYmmdd_HHMMSS.jsonl --out artifacts/pref/dpo.jsonl
```

## 5) QLoRA DPO 训练

```bash
cd /root/csw_test
python llm/train_dpo_qlora.py \
  --model /root/csw_test/models/Qwen2.5-7B-Instruct \
  --data artifacts/pref/dpo.jsonl \
  --out artifacts/pref/dpo_ckpt
```

