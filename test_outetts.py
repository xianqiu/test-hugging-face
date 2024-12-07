from pathlib import Path

import outetts
from _utils import ModelManager


class OuteTTS(object):

    def __init__(self, language, speaker_name=None):
        # Supported languages in v0.2: en, zh, ja, ko
        self._model_name = ModelManager.get("OuteAI/OuteTTS-0.2-500M")
        self._wavtokenizer_name="wavtokenizer_large_speech_320_24k.ckpt"
        self._language = language
        self._device = 'cuda:0'
        self._model = self._set_model()
        self._speaker = self._load_speaker(speaker_name)

    def _load_speaker(self, speaker_name):
            if not speaker_name:
                return
            filepath = Path(f"{speaker_name}.json").resolve()
            print(filepath)
            if filepath.exists():
                speaker = self._model.load_speaker(str(filepath))
            else:
                speaker = self._model.load_default_speaker(name=speaker_name)
            return speaker

    def _set_model(self):
        # Configure the model
        model_config = outetts.HFModelConfig_v1(
            model_path=self._model_name,
            tokenizer_path=self._model_name,
            wavtokenizer_model_path=Path(self._model_name) / self._wavtokenizer_name,
            language=self._language,
            device=self._device
        )
        # Initialize the interface
        return outetts.InterfaceHF(model_version="0.2", cfg=model_config)

    def list_speakers(self):
        self._model.print_default_speakers()

    def generate(self, text):
        output = self._model.generate(
            text=text,
            # Lower temperature values may result in a more stable tone,
            # while higher values can introduce varied and expressive speech
            temperature=0.1,
            repetition_penalty=1.1,
            max_length=4096,
            # Optional: Use a speaker profile for consistent voice characteristics
            # Without a speaker profile, the model will generate a voice with random characteristics
            speaker=self._speaker,
        )
        # Optional: Play the synthesized speech
        # output.play()
        # Save the synthesized speech to a file
        output.save("outputs/outetts_out_text.wav")

    def create_speaker(self, audio_path, transcript_path, speaker_name, save_to):
        #  Create a speaker profile (use a 10-15 second audio clip)
        with open(transcript_path, 'r', encoding='utf-8') as file:
            transcript = file.read()
        speaker = self._model.create_speaker(audio_path=audio_path,
                                             transcript=transcript)
        filepath = Path(save_to) / f"{speaker_name}.json"
        self._model.save_speaker(speaker, filepath)
        print(f">> Speaker [{speaker_name}] created. Profile saved to path [{filepath}] ")


def test_tts_english():
    text = """Speech synthesis is the artificial production of human speech. 
               A computer system used for this purpose is called a speech synthesizer, 
               and it can be implemented in software or hardware products."""
    ot = OuteTTS(language='en', speaker_name='female_1')
    ot.generate(text)


def test_tts_chinese(speaker_name='female_1'):
    text = """你好啊，很高兴见到你。"""
    ot = OuteTTS(language='zh', speaker_name=speaker_name)
    ot.generate(text)


def create_my_speaker(speaker_name):
    ot = OuteTTS(language='zh')
    ot.create_speaker(
        audio_path=f"data/speech/14.wav",
        transcript_path=f"data/speech/14.TXT",
        speaker_name=speaker_name,
        save_to=f"outputs"
    )


if __name__ == '__main__':

    # test_tts_english()
    # test_tts_chinese()
    create_my_speaker("my_speaker")
    test_tts_chinese(speaker_name='outputs/my_speaker')



