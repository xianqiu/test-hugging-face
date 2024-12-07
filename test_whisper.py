from transformers import pipeline
from _utils import ModelManager


def test_asr_chinese():
    model_name = ModelManager.get("openai/whisper-tiny")
    pipe = pipeline(task='automatic-speech-recognition',
                    model=model_name,
                    device='cuda:0')
    filepaths = ['data/asr/12.wav', 'data/asr/13.wav', 'data/asr/14.wav']
    for out in pipe(filepaths):
        print(out['text'])


if __name__ == '__main__':
    test_asr_chinese()

