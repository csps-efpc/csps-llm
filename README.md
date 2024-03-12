# csps-llm
A base for Government of Canada LLM services like chatbots, RAGs, and batch processors. This is the engine behind the CSPS' "Whisper" and "Chuchotte" demo services.

### Getting Started

Given a base install of both Python 3 and `pip`, install dependencies with:
```
pip install flask simple_websocket traceback llama_cpp feedparser requests json bs4
```

You'll need to download a model to use. We recommend one of the Q4 versions of https://huggingface.co/tsunemoto/bagel-dpo-7b-v0.4-GGUF/tree/main
