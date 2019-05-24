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


if __name__ == '__main__':
    df = pd.read_csv('/midata/manceps/unredact/mueller-report-factbase-with-redactions-marked.csv', header=1)
    r = get_probable_redactions(df)
    print(r)
    print()
    print(REDACTION_MARKERS)
