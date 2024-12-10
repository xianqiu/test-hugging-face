from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from _utils import ModelManager


class Llama(object):

    def __init__(self):
        self._model_name = ModelManager.get("meta-llama/Llama-3.2-1B-Instruct")
        self._device = "cuda:0"
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name,
                                                           attn_implementation='eager').to(device=self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    def _format_inputs(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return self._tokenizer(text, return_tensors="pt").to(self._device)

    def generate(self, prompt):
        model_inputs = self._format_inputs(prompt)
        streamer = TextStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        self._model.generate(
            **model_inputs,
            max_new_tokens=512,
            streamer=streamer,
        )


if __name__ == '__main__':
    Llama().generate("用Python实现快速排序")
