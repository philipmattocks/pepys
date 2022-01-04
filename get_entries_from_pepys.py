import re
from urllib import request
from nltk import word_tokenize
from nltk.util import ngrams
import pandas as pd
import logging
from datetime import datetime

def get_ngrams(text,n):
    n_gram_list = ngrams(word_tokenize(text), n)
    return [' '.join(grams) for grams in n_gram_list ]

def set_up_logging(level):
    lev_dict = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO}
    log = logging.getLogger(__name__)
    log.setLevel(lev_dict[level])
    # create console handler and set level to debug
    # ch = logging.FileHandler(filename='log.log', mode='w')
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to log
    log.addHandler(ch)
    return log

def process_raw_text(text):
    logger.info('removing non-original text')
    text = text[text.find('JANUARY 1659-1660'):text.rfind('END OF THE DIARY.')]
    text = re.sub("(\[(?:\[??[^\[]*?\]))", "", text) # remove everything between square brackets
    return text

def find_proper_nouns(tokens):
    names = set()
    for i, t in enumerate(tokens):
        if t.lower() in ['to', 'at', 'in'] \
                and (tokens[i+1][0] == tokens[i+1][0].upper() or  t == 'the') \
                and tokens[i+1] not in ['Mr','Mrs', 'Sir', 'Dr', 'Almighty', 'God', 'English', 'Captain'] \
                and not tokens[i+1][0].isnumeric() \
                and not re.match('L\d+', tokens[i+1]):
            names.add(' '.join([t, tokens[i+1], tokens[i+2], tokens[i+3]]))
        #if t[0] == t[0].upper() and not t[0].isnumeric() and not re.match('L\d+',t):
        # names.add(' '.join([t[i],t[i]]))
    return names

if __name__ == "__main__":
    logger = set_up_logging('INFO')
    logger.info('Logging at INFO')
    logger.debug('Logging at DEBUG')
    MONTHS = {'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4, 'MAY': 5, 'JUNE': 6, 'JULY': 7, 'AUGUST': 8,
              'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12}

    #logger.info('reading text file')
    #with open('./text_files/pepys.txt') as f:
        #raw = f.read()

    url = "https://www.gutenberg.org/files/4200/4200-0.txt"
    response = request.urlopen(url)
    raw = response.read().decode('utf8')
    raw = process_raw_text(raw)
    lines = raw.splitlines()
    entries = []

    for i,l in enumerate(lines):
        date_match = re.match('^([A-Z]+)\s*(?:1\d\d\d-)*(1\d\d\d)$',l)
        if date_match:

            month = date_match.group(1)
            year = date_match.group(2)
            for j,l2 in enumerate(lines[i+1:]):
                if re.match('^([A-Z]+)\s*(?:1\d\d\d-)*(1\d\d\d)$',l2):
                    break
                day_match = re.match('^(?:[a-zA-Z]{3,}\.*\s)*(\d{1,2})(?:th|rd|st|d|nd)\.*\s[A-Z\(]', l2)
                if day_match:
                    entry_text = []
                    day = day_match.group(1)
                    lines_for_entry = []
                    for k in lines[i+j+1:]:
                        day_match2 = re.match('^(?:[a-zA-Z]{3,}\s)*(\d{1,2})(?:th|rd|st|d|nd)\.*\s[A-Z]', k)
                        if day_match2 and day != day_match2.group(1):
                            break
                        if re.search('\w',k):
                            lines_for_entry.append(k)
                            date_obj = datetime(int(year), int(MONTHS[month]), int(day))
                            date_string = date_obj.strftime("%Y-%m-%d")
                            logger.debug(f'adding entry for: {date_string}')
                    entries.append({'date':date_string, 'entry':' '.join(lines_for_entry)})
    logger.info(f'Number of entries: {len(entries)}')
    logger.info(f'Saving to entries.csv')
    pd.DataFrame(entries).to_csv('entries.csv', index=False)



