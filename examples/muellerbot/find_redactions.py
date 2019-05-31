import re
import pandas as pd
import sys
# import keras_bert


def find_redactions(s):
    return re.findall(r'\[[^\]]*\]', s)


def get_probable_redactions(df):
    possible_redactions = set()
    for s in df.Text:
        possible_redactions = possible_redactions.union(set(find_redactions(s)))

    probable_redactions = set()
    for r in probable_redactions:
        if 'grand jury' in r.lower() or 'harm' in r.lower() or 'priva' in r.lower():
            probable_redactions.add(r)
    return probable_redactions


REDACTION_MARKERS = set([
    '[Harm to Ongoing Matter - Grand Jury]',
    '[Harm to Ongoing Matter - Personal Privacy]',
    '[Harm to Ongoing Matter]',
    '[Personal Privacy - Grand Jury]',
    '[Personal Privacy]',
    '[HOM]',
    '[Investigative Technique]',
    '[Investigative Technique]',
    '[IT]',
    'unk'
])

MARKER = 'unk'


def guess_redaction_markers():
    df = pd.read_csv('mueller-report-factbase-with-redactions-marked.csv', header=1)
    r = get_probable_redactions(df)
    print(r)
    print()
    print(REDACTION_MARKERS)


def normalize_redaction_markers(lines, inplace=True):
    normalized_lines = [''] * len(lines) if not inplace else lines
    normalizer = dict(zip(REDACTION_MARKERS, ['__' + x.replace(' ', '_').replace('-', '_')[1:-1] + '__' for x in REDACTION_MARKERS]))
    for i, line in enumerate(lines):
        for k, v in normalizer.items():
            normalized_lines[i] = line.replace(k, v)
    return lines


def clean_dataframe(filepath='mueller-report-with-redactions-marked.csv'):
    df = pd.read_csv(filepath, header=1)
    df.columns = 'page text appendix unnamed'.split()
    return df


def get_line_pairs(df, redaction_marker='[Harm to Ongoing Matter]',
                   min_line_length=40, max_line_length=120):
    line_context = get_line_context(
        df=df, redaction_marker=redaction_marker,
        min_line_length=min_line_length, max_line_length=min_line_length)
    return [tuple(lc[:2]) for lc in line_context]


def get_line_context(df, redaction_marker='[Harm to Ongoing Matter]',
                     min_line_length=40, max_line_length=120):
    line_pairs = []
    for i, line in enumerate(df.text):
        if redaction_marker in line:
            prevline = df.text.iloc[i - 1]
            nextline = df.text.iloc[i + 1] if (i < (len(df) - 1)) else ''
            if (redaction_marker not in prevline and
                    len(prevline) > min_line_length and
                    len(prevline) < max_line_length):
                line_pairs.append((prevline, line, nextline))
    return line_pairs


def find_text(df='mueller-report-with-redactions-marked.csv', substring='of documents and', marker='[Personal Privacy]'):
    df = clean_dataframe(df) if isinstance(df, str) else df
    text = ''
    for t in df.text:
        if substring in t:
            text = t
            break
    # print(f'TEXT: {text}')
    marker_start = text.find(marker)
    marker_stop = marker_start + len(marker)
    # print(f'marker_start: {marker_start}, marker_stop: {marker_stop}')
    prefix = text[:marker_start]
    suffix = text[marker_stop:]
    # print(f'HOM prefix: {prefix}\nHOM suffix: {suffix}')

    return prefix, suffix


def find_repeated_substring(text, substring=MARKER, max_occurences=32):
    """ Find contiguous redaction markers and return the start locations

    >>> text = 'Mueller said "MASK MASK MASK", then walked away.'
    >>> find_repeated_substring(text, 'MASK')
    [14, 19, 24]
    >>> find_repeated_substring('unkunkunk')
    [0, 3, 6]
    >>> find_repeated_substring(' unkunkunk')
    [1, 4, 7]
    >>> find_repeated_substring(' unkunkunk ')
    [1, 4, 7]
    >>> find_repeated_substring('unredact unk if you can.')  # FIXME: shoudl be [1, 4, 8]?
    [9]
    >>> find_repeated_substring(' unkunk unk ')  # FIXME: shoudl be [1, 4, 8]?
    [1, 4, 8]
    """
    # print(f'TEXT: {text}')
    substring = substring or MARKER
    start = text.find(substring)
    stop = start + len(substring)
    starts = []
    for i in range(max_occurences):
        if not (start > -1 and stop <= len(text) - len(substring) + 1):
            break
        # print(start, stop)
        if len(starts):
            stop = starts[-1] + len(substring)
            starts.append(stop + start)
        else:
            starts = [start]
        # print(start, stop)
        start = text[stop:].find(substring)
        if start < 0 and len(starts) > 1:
            return starts[:-1]
        # print(start, stop)
        # print(starts)
    return starts


def main():
    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
    else:
        csv_filename = 'mueller-report-with-redactions-marked.csv'
    df = clean_dataframe(csv_filename)
    line_pairs = get_line_pairs(df)
    line_pairs = pd.DataFrame(line_pairs, columns='line1 line2'.split())
    line_pairs.to_csv('mueller-report-redaction-linepairs.csv')
    print(line_pairs.head())

    # lines = normalize_redaction_markers(df['text'])
