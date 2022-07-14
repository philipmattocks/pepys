import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from flair.models import TextClassifier
from flair.data import Sentence
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk import ngrams

def show_days_with_missing_enties(df):
    """
    Some days are missing from the DF ie when Pepys didn't make an entry, this adds those days in with None in the entry column
    """
    df['date_dt'] = df['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    date_generated = [(df['date_dt'].min() + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(0, (df['date_dt'].max()-df['date_dt'].min()).days)]
    df_filled = pd.DataFrame(date_generated,columns=['date'])
    df = df_filled.merge(df,on='date',how='left')[['date', 'entry']]
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    return df

def length_of_entries(df):
    df = df.loc[df['entry'].isna() == False]
    df['entry_length'] = df['entry'].str.len()
    df['sma_length'] = df['entry_length'].rolling(200).mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    # Specify how our lines should look
    ax.plot(df.date, df.sma_length, alpha=1,label='clah')
    plt.title('200 SMA of char length of entry')
    plt.show()

def basic_facts(df):
    # Missing entries
    basic_facts = {}
    df['missing_entry'] = df['entry'].isna()==True
    prop_with_entries = len(df.loc[df['missing_entry']==False])/len(df)
    #print(f'Proportion of days with entries: {prop_with_entries}')
    basic_facts['prop_days_with_entries'] = prop_with_entries
    fig, ax = plt.subplots(figsize=(10, 6))
    # Specify how our lines should look
    ax.bar(df.date, df.missing_entry)
    plt.title('days with missing entries')
    plt.show()
    # Length of entries
    return basic_facts


def freq_dist(df):
    remerged = ' '.join(df_raw['entry'])
    stop = stopwords.words('english')
    words = word_tokenize(remerged)
    words = [w for w in words if w.lower() not in stop and w.isalpha()]
    ngrams2 = ngrams(words,2)
    fd = FreqDist(words)
    fd2 = FreqDist(ngrams2)
    return fd, fd2


if __name__ == "__main__":
    df_raw = pd.read_csv('entries.csv')
    df_dates_filled = show_days_with_missing_enties(df_raw)
    length_of_entries(df_dates_filled)
    basic_facts_dict = basic_facts(df_dates_filled)
    print(basic_facts_dict)

    fd,fd2 = freq_dist(df_raw)
    print(fd.most_common(20))
    print(fd2.most_common(20))


