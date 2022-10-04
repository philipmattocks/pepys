import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import contractions
from nltk.stem import WordNetLemmatizer


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def expand_contractions(text):
    split_list = [contractions.fix(word) for word in text.split()]
    return ' '.join(split_list)


def tokenise(text):
    return WordPunctTokenizer().tokenize(text)

def tokenise_to_sentences(text):
    return sent_tokenize(text)

def split_on_commas(text):
    return text.split(',')

def remove_stop_words(text, stop_words):
    return [word for word in text if
            word not in (stop_words)]


def remove_non_alpha_numeric(text):
    return [word for word in text if word.isalpha()]


def lower_case(text):
    return text.lower()


def remove_parentheses(text):
    return ''.join([char for char in text if char not in ['(', ')']])

def remove_punct(text):
    return [word for word in text if word not in string.punctuation]


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def get_word_net_tags(text_tags):
    return [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in text_tags]


def lemmatise(pos_tagged_text):
    return [WordNetLemmatizer().lemmatize(tagged_word[0], tagged_word[1]) for tagged_word in pos_tagged_text]


def pre_process(df, remove_contractions=True, make_lower_case=True, remove_parenth=True, split_into_sentences=True,
                split_between_commas=True, remove_punct_from_words=True, remove_stops=True, lematise_words=True):
    df['processed'] = df['entry']
    if remove_contractions:
        df['processed'] = df['entry'].apply(expand_contractions)
    if make_lower_case:
        df['processed'] = df['processed'].apply(lower_case)
    if remove_parenth:
        df['processed'] = df['processed'].apply(remove_parentheses)
    if split_into_sentences:
        df['processed'] = df['processed'].apply(tokenise_to_sentences)
        df = df.explode('processed', ignore_index=True)
    if split_between_commas:
        df['processed'] = df['processed'].apply(split_on_commas)
        df = df.explode('processed', ignore_index=True)
    df['processed'] = df['processed'].apply(tokenise)
    if remove_punct_from_words:
        df['processed'] = df['processed'].apply(remove_punct)
    if remove_stops:
        df['processed'] = df['processed'].apply(lambda x: remove_stop_words(x, stopwords.words('english')))
    if lematise_words:
        df['pos_tags'] = df['processed'].apply(pos_tag)
        df['wordnet_pos_tags'] = df['pos_tags'].apply(get_word_net_tags)
        df['lem'] = df['wordnet_pos_tags'].apply(lemmatise)
        df['lem_joined'] = [' '.join(map(str, letter)) for letter in df['lem']]
    return df


