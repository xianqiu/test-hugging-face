import ChatTTS
import torch
import scipy

from _utils import ModelManager


class TestChatTTS(object):

    def __init__(self):
        self._model_name = ModelManager.get("chattts")
        self._device = 'cuda:0'
        self._model = self._set_model()
        self._sample_rate = 24_000

    def _set_model(self):
        model = ChatTTS.Chat()
        model.load(source='custom',
                   custom_path=self._model_name,
                   compile=False,
                   device=torch.device(self._device))
        return model

    @property
    def _params_refine_text(self):
        # use oral_(0-9), laugh_(0-2), break_(0-7)
        # to generate special token in text to synthesize.
        return ChatTTS.Chat.RefineTextParams(
            prompt='[oral_2][laugh_0][break_6]',
        )

    @property
    def _params_infer_code(self):
        return ChatTTS.Chat.InferCodeParams(
            temperature=0.3,  # using custom temperature
            top_P=0.7,  # top P decode
            top_K=20,  # top K decode
        )

    def generate(self, texts):

        outputs = self._model.infer(
            texts,
            params_refine_text=self._params_refine_text,
            params_infer_code=self._params_infer_code,
        )

        filename = f"outputs/chattts_out.wav"
        scipy.io.wavfile.write(filename,
                               rate=self._sample_rate,
                               data=outputs[0])


def test_tts_chinese():
    tct = TestChatTTS()
    tct.generate(["你好呀, 你是谁，很高兴见到你"])


if __name__ == '__main__':
    test_tts_chinese()




