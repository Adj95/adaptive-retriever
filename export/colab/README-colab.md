```bash
!pip install -r requirements-colab.txt
# upload requests.jsonl
!python generate_offline.py --requests requests.jsonl --results results.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct --batch-size 32 --max-new-tokens 96
# download results.jsonl
```
