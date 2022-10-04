from langdetect import detect
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
from pre_process import pre_process
from google.cloud import translate_v2 as translate
import os

def get_language_for_each_sentence(df, min_words=10):
    df = df[df["processed"].apply(lambda x: len(x) > min_words)]
    df['joined'] = [' '.join(map(str, letter)) for letter in df['processed']]
    df['language'] = df['joined'].apply(detect)
    return df


def get_google_api_results_in_chunks(in_csv, out_csv, chunksize=10):
    with open(in_csv) as f:
        headers = f.readline()
    if not os.path.exists(out_csv):
        with open(out_csv, 'w') as f:
            f.write(headers)
    number_lines = sum(1 for row in (open(in_csv)))

    # start looping through data writing it to a new file for each chunk
    for i in range(1, number_lines, chunksize):
        df = pd.read_csv(in_csv,
                         header=None,
                         nrows=chunksize,  # number of rows to read at each loop
                         skiprows=i)  # skip rows that have been read
        df.columns = headers.strip().split(',')
        df = pre_process(df, lematise_words=False, remove_stops=False, remove_punct_from_words=False)
        df['joined'] = [' '.join(map(str, letter)) for letter in df['processed']]
        df['language'] = df['joined'].apply(detect)
        # df = df.groupby('date', axis=0, as_index=False).agg(
        #     {'entry': ' '.join, 'entry': ' '.join, 'language': list, })


        df.to_csv(out_csv,
                  index=False,
                  header=False,
                  mode='a',  # append data to csv file
                  chunksize=chunksize)  # size of data to append for each loop

if __name__ == "__main__":
    get_google_api_results_in_chunks('entries.csv', 'entries_with_languages.csv')
    # Need to enable google translate API and setup/download json key with translate api access and point to it via:
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/philipmattocks/Work/pepys/tflbikeuse-1a3db08091a5.json"
    # # Specify project like:
    # parent = 'projects/tflbikeuse'
    # client = translate.Client()
    # # client.detect_language('γεια σου, τι κάνεις; ολά καλά;')
    # entries = pd.read_csv('entries.csv')
    # entries = pre_process(entries, lematise_words=False, remove_stops=False, remove_punct_from_words=False)
    # entries = get_language_for_each_sentence(entries)
    # print(l)