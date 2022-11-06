import re
from urllib import request
import pandas as pd
from datetime import datetime

MONTHS = {'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4, 'MAY': 5, 'JUNE': 6, 'JULY': 7, 'AUGUST': 8,
          'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12}
day_formats = ['1st', '2nd', '2d', '3rd', '3d', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th',
               '13th', '14th', '15th', '16th', '17th', '18th', '19th', '20th', '21st', '21St', '21 st', '22nd',
               '22d', '23d', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th', '31st']
DOWNLOAD = False


def remove_text_pepys_did_not_write(text):
    """remove editors' notes etc to leave only Pepys' original text"""
    print('removing non-original text')
    text = text[text.find('JANUARY 1659-1660'):text.rfind('END OF THE DIARY.')]
    editors_notes = '\s\s\s\s\sETEXT EDITOR[\s\S]*?(?=JANUARY)'
    text = re.sub(editors_notes, "", text)
    square_bracket_comments_inline = '(--\[[^]]+\])'
    text = re.sub(square_bracket_comments_inline, "", text)
    square_bracket_paragraphs = '\s\s\s\s\s\[[^]]+\]'
    text = re.sub(square_bracket_paragraphs, "", text)
    with open('original_text_only.txt','w') as f:
        f.write(text)
    return text


def get_structured_data(raw_text, day_formats_list):
    """Convert raw text into pandas DF with date column and text column"""
    lines = raw_text.splitlines()
    entries = []
    # Find dates like 'FEBRUARY 1665-1666' and 'JULY 1661'
    month_year_match_regex = '^([A-Z]+)\s*(?:1\d\d\d-)*(1\d\d\d|\d\d)$'
    # Find days of month, '1st' '23rd' '30th' etc, Pepys formats these dates inconsistently,
    # specified in day_formats_list variable:
    day_of_month_regex = f'^(?:[A-Z]{{1}}[a-zA-Z]{{2,}}\.*\s){{0,1}}({"|".join(day_formats_list)}' \
                         f')(?:\.*\s|,|,\s|\:\s|\.\s\[)[A-Z\(\d]'
    for i, line in enumerate(lines):
        month_year = re.match(month_year_match_regex, line)
        if month_year:
            month = month_year.group(1)
            year = month_year.group(2)
            for j, line_in_month in enumerate(lines[i+1:]):
                if re.match(month_year_match_regex, line_in_month):
                    break
                day_match = re.match(day_of_month_regex, line_in_month)
                if day_match:
                    day = day_match.group(1)
                    day = int(re.search(r'\d+', day).group())
                    lines_for_entry = []
                    for entry_lines in lines[i+j+1:]:
                        day_match2 = re.match(day_of_month_regex, entry_lines)
                        if day_match2 and day != int(re.search(r'\d+', day_match2.group(1)).group()):
                            break
                        if re.search('\w', entry_lines) and not re.match('^([A-Z]+)\s*(?:1\d\d\d-)*(1\d\d\d|\d\d)$',
                                                                         entry_lines):
                            lines_for_entry.append(entry_lines)
                            date_obj = datetime(int(year), int(MONTHS[month]), int(day))
                            date_string = date_obj.strftime("%Y-%m-%d")
                            # some dates only have the last 2 digits, this fixes those eg 67 -> 1667:
                            if len(year) == 2:
                                year = '16' + year
                    entries.append({'date': date_string, 'entry': ' '.join(lines_for_entry)})
    return pd.DataFrame(entries)


def correct_date_typos(df):
    """Correct specific typos in the dates in the raw text"""
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
    entries_df = get_structured_data(raw, day_formats)
    print(f'Number of entries: {entries_df.shape[0]}')
    print(f'correcting typos in dates')
    entries_df = correct_date_typos(entries_df)
    print(f'Saving to entries.csv')
    entries_df.to_csv('entries.csv', index=False)



