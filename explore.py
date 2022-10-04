import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk import ngrams
import nltk
import seaborn as sns
import numpy as np
import string
nltk.download('punkt')
nltk.download('stopwords')

def show_days_with_missing_enties(df):
    """
    Some days are missing from the DF ie when Pepys didn't make an entry, this adds those days in with None in the entry column
    """
    df['date_dt'] = df['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    date_generated = [(df['date_dt'].min() + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(0, (df['date_dt'].max()-df['date_dt'].min()).days)]
    df_filled = pd.DataFrame(date_generated,columns=['date'])
    df = df_filled.merge(df, on='date', how='left')[['date', 'entry']]
    df['missing_entry'] = df['entry'].isna() == True
    return df


def create_dated_heatmap(df, column_name, agg=np.sum):
    entries['date_dt'] = entries['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    df['year'] = df['date_dt'].apply(lambda x: x.year)
    entries['month'] = entries['date_dt'].apply(lambda x: x.month)
    entries['day'] = entries['date_dt'].apply(lambda x: x.day)
    months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
              11: 'Nov', 12: 'Dec'}
    df['month'] = df['month'].replace(months)
    df_piv = df.pivot_table(index='year', columns='month', values=column_name, aggfunc=agg).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df_piv, annot=True, linewidths=1, linecolor='white', ax=ax, cmap="BuPu")
    plt.show()


def freq_dist(df, n):
    remerged = ' '.join(df['entry'])
    stop = stopwords.words('english')
    words = word_tokenize(remerged)
    words = [w for w in words if w.lower() not in stop and w not in string.punctuation + '’‘']
    ngrams_dist = ngrams(words, n)
    return FreqDist(ngrams_dist)


if __name__ == "__main__":
    entries_raw = pd.read_csv('entries.csv')
    entries = show_days_with_missing_enties(entries_raw)
    prop_with_entries = len(entries.loc[entries['missing_entry'] == False]) / len(entries)
    print(f'Proportion of days with entries: {prop_with_entries}')
    create_dated_heatmap(entries, 'missing_entry')
    entries['missing_entry'] = entries['entry'].isna() == True
    fd1 = freq_dist(entries_raw, 1)
    # fd2 = freq_dist(entries, 2)
    # fd3 = freq_dist(entries, 3)
    fd1.plot(25, title='Freq dist of top 25 words')
    # fd2.plot(25, title='Freq dist of top 25 2grams')
    # fd3.plot(25, title='Freq dist of top 25 3grams')


