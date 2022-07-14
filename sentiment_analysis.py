from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords

def flair_prediction(x, sia):
    sentence = Sentence(x)
    sia.predict(sentence)
    return sentence.labels[0].score

def get_sentiment(df, entry_column, sentiment_col_name):
    sia = TextClassifier.load('en-sentiment')
    df[sentiment_col_name] = df[entry_column].apply(lambda x: flair_prediction(x, sia))
    sentence = Sentence(df[entry_column][0])
    sia.predict(sentence)
    score = sentence.labels[0].score
    # print(score)
    return df

def process_text_for_sentiment(entries_df):
    # Remove stop words and non-alpha characters for sentiment analysis
    entries_df['processed_entry'] = entries_df['entry'].apply(lambda x: ' '.join([word for word in WordPunctTokenizer().tokenize(x) if word.isalpha() and word not in (stop) or word == '.']))
    return entries_df

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def char_in_string(string1, string2):
    "Check if any characters from string1 are in string2"
    return any([i for i in string1 if i in string2 ])


if __name__ == "__main__":

    df_raw = pd.read_csv('entries.csv')
    df_raw = df_raw[:100]
    stop = stopwords.words('english')
    df = process_text_for_sentiment(df_raw)
    #df = get_sentiment(df_raw,'entry', 'sentiment_raw')
    df = get_sentiment(df, 'processed_entry', 'sentiment')
    # sentiment.to_csv('entries_with_sentiment.csv', index=False)
    # # df = pd.read_csv('entries_with_sentiment.csv')
    fig, ax = plt.subplots(figsize=(10, 6))
    df.hist(ax=ax,column='sentiment')
    plt.show()
    # #
    # #
    # #
    df['sentiment_sma'] = df['sentiment'].rolling(5).mean()
    # # df['sentiment_sma_no_stop'] = df['sentiment'].rolling(200).mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    # # # Specify how our lines should look
    ax.plot(df['date'], df['sentiment_sma'], alpha=1,label='blah')
    plt.title('SMA of sentiment per entry')
    plt.show()