### 离线模型

#### 命名规范

* 模型名: "Qwen/Qwen2.5-0.5B-Instruct"
* 模型对应的文件夹名: "models--Qwen--Qwen2.5-0.5B-Instruct"
  * 添加前缀: models--
  * 符号替换: "/" --> "--" 

#### 使用方法

1. 如果模型已经下载，进入 huggingface 的缓存文件夹，把模型对应的文件夹拷贝到 `models/`。说明：Windows 下的缓存地址是：
`C:\Users\<username>\.cache\huggingface\hub`。
2. 如果模型没有下载，可以用 `test-hugging-face/_utils.py` 中的 `ModelManager.download(model_name)`下载模型。说明：
它先把模型下载到默认的缓存，然后移动到 `models/`。
3. 使用 `test-hugging-face/_utils.py` 中的 `ModelManager.get(model_name)`, e.g,
   ```python
   from transformers import AutoModelForCausalLM
   from _utils import ModelManager
   
   model_name = ModelManager.get("Qwen/Qwen2.5-0.5B-Instruct")
   model = AutoModelForCausalLM.from_pretrained(model_name)
   ```


