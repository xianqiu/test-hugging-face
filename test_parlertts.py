from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

import soundfile as sf
import sounddevice as sd

from _utils import ModelManager


class ParlerTTS(object):

    def __init__(self):
        self._device = "cuda:0"
        self._model_name = ModelManager.get("parler-tts/parler-tts-mini-v1")
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(self._model_name,
                                                            attn_implementation='eager').to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    def generate(self, text, description=None):
        input_ids = self._tokenizer(description, return_tensors="pt").input_ids.to(self._device)
        prompt_input_ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._device)

        generation = self._model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write("outputs/parler_tts_out.wav", audio_arr, self._model.config.sampling_rate)
        sd.play(audio_arr, self._model.config.sampling_rate)
        sd.wait()


def test_tts_english():
    text = "Hey, how are you doing today?"
    description = """ A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. 
        The recording is of very high quality, with the speaker's voice sounding clear and very close up.
        """
    ParlerTTS().generate(text, description)


if __name__ == '__main__':
    test_tts_english()