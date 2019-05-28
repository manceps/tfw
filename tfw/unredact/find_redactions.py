import re
import pandas as pd
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
])


def guess_redaction_markers():
    df = pd.read_csv('/midata/manceps/unredact/mueller-report-factbase-with-redactions-marked.csv', header=1)
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


if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
    else:
    	csv_filename = '/midata/manceps/unredact/mueller-report.csv'
    df = pd.read_csv(csv_filename, header=1)
    lines = normalize_redaction_markers(df['text'])
