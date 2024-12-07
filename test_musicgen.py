from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
import numpy as np
import soundfile as sf
import torch

from _utils import ModelManager


class MusicGen(object):

    def __init__(self):
        self._device = 'cuda:0'
        self._model_name = ModelManager.get("facebook/musicgen-small")
        self._model = MusicgenForConditionalGeneration.from_pretrained(self._model_name,
                                                             attn_implementation='eager').to(device=self._device)
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._sampling_rate = self._model.config.audio_encoder.sampling_rate

    def _seconds_to_tokens(self, seconds):
        return seconds * self._model.config.audio_encoder.frame_rate

    def _save(self, mono_audio, prefix=None):
        # 把gpu中的tensor复制到cpu
        if isinstance(mono_audio, torch.Tensor):
            mono_audio = mono_audio.cuda(0).cpu()
        # 单声道 => 双声道
        music = np.column_stack((mono_audio, mono_audio))
        filename = f"outputs/musicgen_out_{prefix}.wav"
        scipy.io.wavfile.write(filename,
                               rate=self._sampling_rate,
                               data=music)

    def generate_at_random(self, seconds):
        unconditional_inputs = self._model.get_unconditional_inputs(num_samples=1)
        tokens = self._seconds_to_tokens(seconds)
        # output shape = (batch_size, num_channels, sequence_length)
        audio_values = self._model.generate(**unconditional_inputs,
                                            do_sample=True,
                                            max_new_tokens=tokens)
        self._save(audio_values[0, 0], prefix='random')

    def generate_by_text(self, seconds, texts):
        inputs = self._processor(
            text=texts,
            padding=True,
            return_tensors="pt",
        )
        tokens = self._seconds_to_tokens(seconds)
        audio_values = self._model.generate(**inputs.to(self._device),
                                            do_sample=True,
                                            guidance_scale=3,
                                            max_new_tokens=tokens)
        self._save(audio_values[0, 0], prefix='text')

    @staticmethod
    def _load_audio_files(audio_files):
        if not isinstance(audio_files, list):
            audio_files = [audio_files]
        audios = []
        for filename in audio_files:
            audio, _ = sf.read(filename)
            audios.append(audio[:, 0])
        return audios

    def generate_by_audio_and_text(self, seconds, audio_files, sampling_rate, texts):
        audios = self._load_audio_files(audio_files)
        inputs = self._processor(
            audio=audios,
            sampling_rate=sampling_rate,
            text=texts,
            padding=True,
            return_tensors="pt",
        )
        tokens = self._seconds_to_tokens(seconds) // len(audio_files)
        audio_values = self._model.generate(**inputs.to(self._device),
                                            do_sample=True,
                                            guidance_scale=3,
                                            max_new_tokens=tokens)

        # post-process to remove padding from the batched audio
        audio_values = self._processor.batch_decode(audio_values,
                                                    padding_mask=inputs.padding_mask)
        self._save(audio_values[0][0], prefix='audio_text')


def test_tta_random_music():
    mg = MusicGen()
    mg.generate_at_random(seconds=10)


def test_tta_text_to_music():
    mg = MusicGen()
    texts = ["80s pop track with bassy drums and synth",
             "90s rock song with loud guitars and heavy drums"]
    mg.generate_by_text(seconds=10, texts=texts)


def test_tta_text_and_audio_to_music():
    mg = MusicGen()
    texts = ["80s pop track with bassy drums and synth",
             "90s rock song with loud guitars and heavy drums"]

    audio_files = ['data/audios/a1.wav', 'data/audios/a2.wav']
    mg.generate_by_audio_and_text(seconds=15,
                                  audio_files=audio_files,
                                  sampling_rate=32_000,
                                  texts=texts)


if __name__ == '__main__':
    # test_tta_random_music()
    # test_tta_text_to_music()
    test_tta_text_and_audio_to_music()



