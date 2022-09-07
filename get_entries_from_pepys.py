import re
from urllib import request
from nltk import word_tokenize
from nltk.util import ngrams
import pandas as pd
from datetime import datetime

MONTHS = {'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4, 'MAY': 5, 'JUNE': 6, 'JULY': 7, 'AUGUST': 8,
          'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12}
day_formats = ['1st', '2nd', '2d', '3rd', '3d', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th',
               '13th', '14th', '15th', '16th', '17th', '18th', '19th', '20th', '21st', '21St', '21 st', '22nd',
               '22d', '23d', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th', '31st']
DOWNLOAD = False


def remove_text_pepys_did_not_write(text):
    print('removing non-original text')
    text = text[text.find('JANUARY 1659-1660'):text.rfind('END OF THE DIARY.')]
    editors_notes = '\s\s\s\s\sETEXT EDITOR[\s\S]*?(?=JANUARY)'
    text = re.sub(editors_notes, "", text)
    square_bracket_comments_inline = '(--\[[^]]+\])'
    text = re.sub(square_bracket_comments_inline, "", text)
    square_bracket_paragraphs = '\s\s\s\s\s\[[^]]+\]'
    text = re.sub(square_bracket_paragraphs, "", text)
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


def get_structured_data(raw_text):
    lines = raw_text.splitlines()
    entries = []
    month_year_match_regex = '^([A-Z]+)\s*(?:1\d\d\d-)*(1\d\d\d|\d\d)$'
    day_of_month_regex = f'^(?:[A-Z]{{1}}[a-zA-Z]{{2,}}\.*\s){{0,1}}({"|".join(day_formats)})(?:\.*\s|,|,\s|\:\s|\.\s\[)[A-Z\(\d]'
    for i, line in enumerate(lines):
        month_year = re.match(month_year_match_regex, line)
        if month_year:
            month = month_year.group(1)
            year = month_year.group(2)
            for j, line2 in enumerate(lines[i+1:]):
                if re.match(month_year_match_regex, line2):
                    break
                day_match = re.match(day_of_month_regex, line2)
                if day_match:
                    day = day_match.group(1)
                    day = int(re.search(r'\d+', day).group())
                    lines_for_entry = []
                    for lines3 in lines[i+j+1:]:
                        day_match2 = re.match(day_of_month_regex, lines3)
                        if day_match2 and day != int(re.search(r'\d+', day_match2.group(1)).group()):
                            break
                        if re.search('\w', lines3) and not re.match('^([A-Z]+)\s*(?:1\d\d\d-)*(1\d\d\d|\d\d)$', lines3):
                            lines_for_entry.append(lines3)
                            date_obj = datetime(int(year), int(MONTHS[month]), int(day))
                            date_string = date_obj.strftime("%Y-%m-%d")
                            if len(year) == 2:
                                year = '16' + year
                    entries.append({'date': date_string, 'entry': ' '.join(lines_for_entry)})
    return entries


def correct_date_typos(df):
    dates = df['date']
    dates.iloc[df.loc[df['entry'].str.startswith('4th. At the office all the morning. At noon I')].index] = '1661-11-14'
    dates.iloc[df.loc[df['entry'].str.startswith('10th. Sir W. Pen and I did a little business at')].index] = '1662-05-20'
    dates.iloc[df.loc[
        df['entry'].str.startswith('5th. Up, and in Sir W. Batten’s coach to White')].index] = '1664-12-06'
    df['date'] = dates
    return df[['date', 'entry']]


if __name__ == "__main__":

    if DOWNLOAD:
        url = "https://www.gutenberg.org/files/4200/4200-0.txt"
        print(f'Downloading from {url}')
        response = request.urlopen(url)
        raw = response.read().decode('utf8')
    else:
        print('reading text file')
        with open('./4200-0.txt') as f:
            raw = f.read()

    raw = remove_text_pepys_did_not_write(raw)
    entries_data = get_structured_data(raw)
    print(f'Number of entries: {len(entries_data)}')
    entries_df = pd.DataFrame(entries_data)
    entries_df['dup_date'] = entries_df['date'].duplicated()
    duplicate_dates = entries_df.loc[entries_df['dup_date'] == True]['date']
    entries_with_dups = entries_df.loc[entries_df['date'].isin(duplicate_dates)]
    print(f'Following entries seem to have duplicates caused by typos in the source text:  {entries_with_dups}')
    print(f'correcting typos in dates')
    entries_df = correct_date_typos(entries_df)
    print(f'Saving to entries.csv')
    entries_df.to_csv('entries.csv', index=False)



