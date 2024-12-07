from transformers import AutoProcessor, AutoModel
import torch
import scipy
import numpy as np

from _utils import ModelManager


class Bark(object):

    def __init__(self):
        self._model_name = ModelManager.get("suno/bark-small")
        self._device = 'cuda:0'
        self._model = AutoModel.from_pretrained(self._model_name).to(device=self._device)
        self._sample_rate = self._model.generation_config.sample_rate
        self._model.enable_cpu_offload()

    def _save(self, mono_audio):
        # 把gpu中的tensor复制到cpu
        if isinstance(mono_audio, torch.Tensor):
            mono_audio = mono_audio.cpu().cpu().numpy().squeeze()
        # 单声道 => 双声道
        music = np.column_stack((mono_audio, mono_audio))
        filename = f"outputs/bark_out.wav"
        scipy.io.wavfile.write(filename,
                               rate=self._sample_rate,
                               data=music)

    def generate(self, text, voice_preset=None):
        processor = AutoProcessor.from_pretrained(self._model_name, voice_preset=voice_preset)
        inputs = processor(
            text=text,
            return_tensors="pt",
        )
        speech_values = self._model.generate(**inputs.to(device=self._device),
                                            do_sample=True)
        self._save(speech_values)


def test_tts_english():
    sb = Bark()
    text_to_speech = """
         Hello, my name is Suno. And, uh — and I like pizza. [laughs]
         But I also have other interests such as playing tic tac toe.
        """
    sb.generate(text_to_speech, voice_preset='v2/en_speaker_6')


def test_tts_chinese():
    sb = Bark()
    text_to_speech_zh = "你好啊, 你叫什么名字? 很高兴见到你!"
    # voice preset cf. https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
    sb.generate(text_to_speech_zh, voice_preset='v2/zh_speaker_9')


def test_tta_music():
    sb = Bark()
    text_to_music = "♪ Hello, my dog is cute ♪"
    sb.generate(text_to_music)


if __name__ == '__main__':
    # test_bark_tts_english()
    # test_bark_tts_chinese()
    test_tta_music()




