### 环境

* 操作系统: Windows 10 企业版
* CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz   4.20 GHz 
* 内存: 64G 
* GPU: GTX-960 显存 4G
* CUDA: 12.6
* Python: 3.12.7
* PyTorch: 2.5.1
* Transformers: 4.46.1

### 模型

| Model Name                    | Tags                 | Source         | File              |
|-------------------------------|----------------------|----------------|-------------------|
| Qwen/Qwen2.5-0.5B-Instruct    | LM / text generation | Alibaba        | test_qwen.py      |
| openai/whisper-tiny           | Audio / asr          | OpenAI         | test_whisper.py   |
| chattts                       | Audio / tts          | 2noise @github | test_chattts.py   |
| OuteAI/OuteTTS-0.2-500M       | Audio / tts          | OuteAI         | test_outetts.py   |
| parler-tts/parler-tts-mini-v1 | Audio / tts          | HuggingFace    | test_parlertts.py |
| facebook/musicgen-small       | Audio / tta          | Meta           | test_musicgen.py  |
| suno/bark-small               | Audio / tta          | Suno           | test_bark.py      |
| facebook/sam2-hiera-large     | CV / mask generation | Suno           | test_bark.py      |

