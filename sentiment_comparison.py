import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from pysentimiento import create_analyzer
from flair.models import TextClassifier
import pandas as pd
from nltk.tokenize import sent_tokenize
from flair.data import Sentence
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

vader_analyser = SentimentIntensityAnalyzer()
huggingface_analyser = create_analyzer(task="sentiment", lang="en")


# flair_analyser = TextClassifier.load('en-sentiment')

def get_vader_sentiment_for_a_sentence(sentence):
    return vader_analyser.polarity_scores(sentence)['compound']


def get_textblob_sentiment_for_a_sentence(sentence):
    return TextBlob(sentence).sentiment.polarity


def get_huggingface_sentiment_for_a_sentence(sentence):
    return sum(np.array([-1, 0, 1]) * np.array(list(huggingface_analyser.predict(sentence).probas.values())))


# def get_flair_sentiment_for_a_sentence(sentence):
#     flair_sentence = Sentence(sentence)
#     flair_analyser.predict(flair_sentence)
#     return flair_sentence.tag


def split_into_sentences(df):
    df['sentences'] = df['entry'].apply(sent_tokenize)
    return df.explode('sentences', ignore_index=True)


def get_date_objects_from_date_string(df, column, format='%Y-%m-%d'):
    df['date'] = df[column].apply(lambda x: datetime.strptime(x, format))
    df['year'] = df[column].apply(lambda x: x.year)
    df['month'] = df[column].apply(lambda x: x.month)
    df['day'] = df[column].apply(lambda x: x.day)
    return df

def produce_histograms(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    df.hist(ax=ax, column='vader_sentiment', color='blue', alpha=0.5,)
    df.hist(ax=ax, column='huggingface_sentiment', color='red', alpha=0.5)
    plt.title('Histogrm of polarities for Vader (blue) and Huggingface (red) ')
    plt.show()

def produce_sma_graphs(df):
    df['vader_sentiment_sma'] = df['vader_sentiment'].rolling(15).mean()
    df['huggingface_sentiment_sma'] = df['huggingface_sentiment'].rolling(15).mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['date'], df['vader_sentiment_sma'], alpha=1, color='blue')
    ax.plot(df['date'], df['huggingface_sentiment_sma'], alpha=1, color='red')
    plt.title('SMA of sentiment per entry in 1666 for Vader (blue) and Huggingface (red)')
    plt.show()
    return df

def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    d = d.reindex(columns=(pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=int, name='month')))
    sns.heatmap(d, **kwargs)

def produce_heatmaps(df):
    # Plot heatmap of sentiment for all years:
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    fg = sns.FacetGrid(df, col='year', col_wrap=1, margin_titles=True, height=10)
    fg.map_dataframe(draw_heatmap, 'month', 'day', 'vader_sentiment',
                     cbar=False, annot=True, linewidths=1, linecolor='white',
                     cmap=cmap, center=0)
    plt.title('Heatmap of sentiment per entry in 1666 for Vader')
    plt.show()
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    fg = sns.FacetGrid(df, col='year', col_wrap=1, margin_titles=True, height=10)
    fg.map_dataframe(draw_heatmap, 'month', 'day', 'huggingface_sentiment',
                     cbar=False, annot=True, linewidths=1, linecolor='white',
                     cmap=cmap, center=0)
    plt.title('Heatmap of sentiment per entry in 1666 for Huggingface')
    plt.show()


if __name__ == "__main__":
    entries = pd.read_csv('entries.csv')
    entries = split_into_sentences(entries)
    entries = get_date_objects_from_date_string(entries, 'date')
    entries = entries.loc[entries['year'] == 1666]

    entries['vader_sentiment'] = entries['sentences'].apply(get_vader_sentiment_for_a_sentence)
    entries['huggingface_sentiment'] = entries['sentences'].apply(get_huggingface_sentiment_for_a_sentence)


    entries = entries.groupby(['date', 'year', 'month', 'day'], axis=0, as_index=False).agg(
        {'entry': ' '.join, 'entry': ' '.join, 'vader_sentiment': 'sum', 'huggingface_sentiment': 'sum'})
    entries['vader_sentiment_norm'] = (entries['vader_sentiment'] - min(entries['vader_sentiment'])) / (
                max(entries['vader_sentiment']) - min(entries['vader_sentiment']))
    entries['huggingface_sentiment_norm'] = (entries['huggingface_sentiment'] - min(
        entries['huggingface_sentiment'])) / (max(entries['huggingface_sentiment']) - min(entries['huggingface_sentiment']))

    produce_histograms(entries)

    produce_sma_graphs(entries)

    produce_heatmaps(entries)
    plt.show()
