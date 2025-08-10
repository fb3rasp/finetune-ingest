Q&A Data Generator

Usage examples:

```bash
python src/main.py \
  --provider local \
  --model qwen3:14b \
  --ollama-base-url http://192.168.50.133:11434 \
  --incoming-dir /data/incoming \
  --process-dir /data/process \
  --output-file /data/results/training_data.json \
  --questions-per-chunk 5 \
  --temperature 0.3 \
  --batch-processing \
  --resume
```
