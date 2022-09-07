from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from datetime import datetime

def get_sentiment_for_each_day(df):

    df['sentences'] = df['entry'].apply(sent_tokenize)
    df = df.explode('sentences', ignore_index=True)
    analyser = SentimentIntensityAnalyzer()
    df['polarity_scores'] = df['sentences'].apply(analyser.polarity_scores)
    df['sentiment'] = df['polarity_scores'].apply(lambda x: x['compound'])
    return df


if __name__ == "__main__":
    entries = pd.read_csv('entries.csv')
    entries = get_sentiment_for_each_day(entries)
    # Use below line if we have created multiple rows for each entry, eg splitting entries into sentences:
    entries = entries.groupby('date', axis=0, as_index=False).agg(
        {'entry': ' '.join, 'entry': ' '.join, 'sentiment': 'mean', })
    entries['date'] = entries['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    top_negative = entries.sort_values('sentiment', ascending=True).head(10)[['date', 'entry', 'sentiment']]
    top_positive = entries.sort_values('sentiment', ascending=False).head(10)[['date', 'entry', 'sentiment']]
    fig, ax = plt.subplots(figsize=(10, 6))
    entries.hist(ax=ax, column='sentiment')
    plt.show()
    entries['year'] = entries['date'].apply(lambda x: x.year)
    entries = entries.loc[entries['year'] == 1666]
    entries['sentiment_sma'] = entries['sentiment'].rolling(15).mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(entries['date'], entries['sentiment_sma'], alpha=1)
    plt.title('SMA of sentiment per entry')
    plt.show()

