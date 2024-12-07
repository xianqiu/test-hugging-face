import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline


def test_sentiment_analysis():
    # The default model is  "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    pipe = pipeline("sentiment-analysis")
    texts = ["We are very happy to show you the 🤗 Transformers library.",
             "We hope you don't hate it."]
    result = pipe(texts)
    print(result)


def test_summarization():
    pipe = pipeline(task="summarization")
    text = """
    The pipelines are a great and easy way to use models for inference. 
    These pipelines are objects that abstract most of the complex code from the library, 
    offering a simple API dedicated to several tasks, 
    including Named Entity Recognition, Masked Language Modeling, 
    Sentiment Analysis, Feature Extraction and Question Answering. 
    """
    result = pipe(text, min_length=5, max_length=30)
    print(result)


if __name__ == '__main__':
    test_sentiment_analysis()
    test_summarization()






