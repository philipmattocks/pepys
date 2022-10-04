from textblob import TextBlob
import pandas as pd
from pre_process import pre_process
import matplotlib.pyplot as plt
from datetime import datetime


def get_sentiment_for_each_day(df):
    df = pre_process(df)
    df['sentiment'] = df['lem_joined'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df


if __name__ == "__main__":
    entries = pd.read_csv('entries.csv')
    entries = get_sentiment_for_each_day(entries)
    entries['date'] = entries['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    entries = entries.sort_values(by='date', ascending=True)
    # Use below line if we have created multiple rows for each entry, eg splitting entries into sentences:
    entries = entries.groupby('date', axis=0, as_index=False).agg({'lem_joined': ' '.join, 'entry': ' '.join, 'sentiment':'mean',})
    entries['year'] = entries['date'].apply(lambda x: x.year)
    top_negative = entries.sort_values('sentiment', ascending=True).head(10)[['date', 'entry', 'sentiment']]
    top_positive = entries.sort_values('sentiment', ascending=False).head(10)[['date', 'entry', 'sentiment']]
    fig, ax = plt.subplots(figsize=(10, 6))
    entries.hist(ax=ax, column='sentiment')
    plt.show()
    entries = entries.loc[entries['year'] == 1666]
    entries['sentiment_sma'] = entries['sentiment'].rolling(5).mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(entries['date'], entries['sentiment_sma'], alpha=1, label='blah')
    plt.title('SMA of sentiment per entry')
    plt.show()
