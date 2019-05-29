import sys
import os
import codecs
import re
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from find_redactions import clean_dataframe, get_line_pairs


REDACTION_MARKERS = [
    '[Harm to Ongoing Matter - Grand Jury]',
    '[Harm to Ongoing Matter - Personal Privacy]',
    '[Harm to Ongoing Matter]',
    '[Personal Privacy - Grand Jury]',
    '[Personal Privacy]',
    '[HOM]',
]

REDACTION_MARKER = REDACTION_MARKERS[2]
REDACTION_BERT_TOKEN = '[MASK]'

UNZIPPED_MODEL_PATH = '~/midata/public/models/bert/uncased_L-12_H-768_A-12'
if len(sys.argv) == 2:
    UNZIPPED_MODEL_PATH = sys.argv[1]

UNZIPPED_MODEL_PATH = (
    os.path.abspath(os.path.expanduser(os.environ.get('UNZIPPED_MODEL_PATH') or UNZIPPED_MODEL_PATH)))

print(f'UNZIPPED_MODEL_PATH: {UNZIPPED_MODEL_PATH}')


BERT_MODEL_CASED = 'uncased' not in UNZIPPED_MODEL_PATH.lower()


def get_unredacted_sentences(df='mueller-report-with-redactions-marked.csv',
                             min_line_length=60, max_line_length=150):
    df = clean_dataframe(df) if isinstance(df, str) else df
    line_pairs = get_line_pairs(df, min_line_length=min_line_length, max_line_length=max_line_length)
    print(pd.DataFrame(line_pairs, columns='text redacted_text'.split()).head())

    for i, (text, redacted) in enumerate(line_pairs):
        if len(re.findall(r'^[-:.0-9 \t]{1,2}', text.strip())) > 0:
            print(f'Skipping: {text[:30]}')
            continue
        print(f"Redacting: {text[:30]}")
        print()


TEXT = '''
    The presidential campaign of Donald J. Trump ("Trump Campaign" or "Campaign") showed
    interest in WikiLeaks\'s releases of documents and welcomed their potential to damage
    candidate Clinton. Beginning in June 2016, [Harm to Ongoing Matter] forecast to senior
    Campaign officials that WikiLeaks would release information damaging to candidate Clinton.
    WikiLeaks\'s first release came in July 2016. Around the same time, candidate Trump announced
    that he hoped Russia would recover emails described as missing from a private server used by
    linton when she was Secretary of State (he later said that he was speaking sarcastically).
    '''


# if len(sys.argv) != 4:
#     print('USAGE: python load_model.py CONFIG_PATH CHECKPOINT_PATH DICT_PATH')
#     print()
#     sys.argv = [
#         sys.argv[0],
#         os.environ.get('CONFIG_PATH') or os.path.join(UNZIPPED_MODEL_PATH, 'bert_config.json'),
#         os.environ.get('CHECKPOINT_PATH') or os.path.join(UNZIPPED_MODEL_PATH, 'bert_model.ckpt'),
#         os.environ.get('DICT_PATH') or os.path.join(UNZIPPED_MODEL_PATH, 'vocab.txt')]
#     SCRIPTNAME, CONFIG_PATH, CHECKPOINT_PATH, DICT_PATH = sys.argv
#     print(f'CONFIG_PATH:     {CONFIG_PATH}')
#     print(f'CHECKPOINT_PATH: {CHECKPOINT_PATH}')
#     print(f'DICT_PATH:       {DICT_PATH}')

# config_path, checkpoint_path, dict_path = tuple(sys.argv[1:])

# model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
# model.summary(line_length=120)

# token_dict = {}
# with codecs.open(dict_path, 'r', 'utf8') as reader:
#     for line in reader:
#         token = line.strip()
#         token_dict[token] = len(token_dict)
# token_dict_rev = {v: k for k, v in token_dict.items()}

# tokenizer = Tokenizer(token_dict)


class NLPPipeline(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            setattr(self, k, v)


def load_pipeline(UNZIPPED_MODEL_PATH=UNZIPPED_MODEL_PATH, cased=BERT_MODEL_CASED):
    if len(sys.argv) != 4:
        print('python load_model.py CONFIG_PATH CHECKPOINT_PATH DICT_PATH')
        print('CONFIG_PATH:     $UNZIPPED_MODEL_PATH/bert_config.json')
        print('CHECKPOINT_PATH: $UNZIPPED_MODEL_PATH/bert_model.ckpt')
        print('DICT_PATH:       $UNZIPPED_MODEL_PATH/vocab.txt')
        sys.argv = [
            sys.argv[0],
            os.path.abspath(os.environ.get('CONFIG_PATH') or os.path.join(UNZIPPED_MODEL_PATH, 'bert_config.json')),
            os.path.abspath(os.environ.get('CHECKPOINT_PATH') or os.path.join(UNZIPPED_MODEL_PATH, 'bert_model.ckpt')),
            os.path.abspath(os.environ.get('DICT_PATH') or os.path.join(UNZIPPED_MODEL_PATH, 'vocab.txt')),
        ]

    print(sys.argv)
    if not all([os.path.exists(p) for p in sys.argv[1:4]]):
        print("You must specify the path where you've downloaded the pretrained BERT model in $UNZIPPED_MODEL_PATH or on the commandline.")

    config_path, checkpoint_path, dict_path = tuple(sys.argv[1:])

    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
    model.summary(line_length=120)

    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    token_dict_rev = {v: k for k, v in token_dict.items()}
    if cased:
        print('***************CASED TOKENIZER*******************')
    else:
        print('***************uncased tokenizer*******************')
    tokenizer = Tokenizer(token_dict, cased=cased)

    return NLPPipeline(model=model, token_dict=token_dict, token_dict_rev=token_dict_rev, tokenizer=tokenizer)


if 'P' not in globals() and 'P' not in locals():
    P = load_pipeline()


def find_first_hom_tokens(df, text=TEXT, substring='of documents and'):
    if not text:
        df = clean_dataframe(df) if isinstance(df, str) else df
        for t in df.text:
            if substring in t:
                text = t
                break
    print(f'TEXT: {text}')
    tokens = P.tokenizer.tokenize(text)
    joined_tokens = ' '.join(tokens)
    print(f'joined_tokens: {joined_tokens}')
    hom = ' '.join(P.tokenizer.tokenize('[Harm to Ongoing Matter]')[1:-1])
    print(f'joined_hom: {hom}')
    hom_start = joined_tokens.find(hom)
    hom_stop = hom_start + len(hom)
    print(f'hom_start: {hom_start}, hom_stop: {hom_stop}')
    prefix_tokens = joined_tokens[:hom_start].split()
    suffix_tokens = joined_tokens[hom_stop:].split()
    print(f'HOM prefix_tokens: {prefix_tokens}\nHOM suffix_tokens: {suffix_tokens}')

    return prefix_tokens, suffix_tokens


sentences = [
    'The IRA later used social media accounts and interest groups to sow discord in the U.S. political system through what it termed "information warfare."',
    ('The campaign evolved from a generalized program designed in 2014 and 2015 to undermine the U.S. electoral system,' +
        'to a targeted operation that by early 2016 favored candidate Trump and disparaged candidate Clinton.'),
    ('The IRA\'s operation also included the purchase of political advertisements on social media in the names of U.S. persons and entities,' +
        'as well as the staging of political rallies inside the United States.'),
    ('To organize those rallies, IRA employees posed as U.S. grassroots entities and persons and made contact' +
        ' with Trump supporters and Trump Campaign officials in the United States.'),
    ]


MASK_TOKEN = '[MASK]'


def unredact_tokens(prefix_tokens=[], suffix_tokens=[], num_redactions=5):
    tokens = list(prefix_tokens) + [MASK_TOKEN] * num_redactions + list(suffix_tokens)
    tokens = tokens[:512]
    tokens_original = tokens.copy()
    text = ' '.join(tokens)
    print(f"Prediction {num_redactions} MASK tokens in: {' '.join(tokens)}")

    indices = np.asarray([[P.token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
    segments = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens))])
    masks = np.asarray([[0] * 512])
    redactions = []
    for i, t in enumerate(tokens):
        if t == MASK_TOKEN:
            redactions.append(i - 1)
            masks[0][i] = 1

    # masks = np.asarray([[0, 1, 1] + [0] * (512 - 3)])

    predicts = P.model.predict([indices, segments, masks])[0]
    predicts = np.argmax(predicts, axis=-1)
    predictions_parameterized = list(
        map(lambda x: P.token_dict_rev[x],
            [x for (j, x) in enumerate(predicts[0]) if j - 1 in redactions])
        )
    # predictions_hardcoded = list(map(lambda x: token_dict_rev[x], predicts[0][3:5]))
    print(f'Predictions: {predictions_parameterized}')

    # print(f'Hardcoded fill with: {predictions_hardcoded}')
    # list(map(lambda x: token_dict_rev[x], predicts[0][1:3]))
    print(f'.    Actual: {[t for (i, t) in enumerate(tokens_original) if i - 1 in redactions]}')
    print()
    print()
    # if len(predictions) > 10:
    #     break
    return (predictions_parameterized, text)


def unredact_text(text, redactions=[2, 3]):
    print(f"Redacting tokens {redactions} in: {text}")

    tokens = P.tokenizer.tokenize(text)
    tokens_original = tokens.copy()
    for r in redactions:
        tokens[r + 1] = MASK_TOKEN

    print(f'Tokens: {tokens}')

    indices = np.asarray([[P.token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
    segments = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens))])
    masks = np.asarray([[0] * 512])
    for r in redactions:
        masks[0][r + 1] = 1
    # masks = np.asarray([[0, 1, 1] + [0] * (512 - 3)])

    predicts = P.model.predict([indices, segments, masks])[0]
    predicts = np.argmax(predicts, axis=-1)
    predictions_parameterized = list(
        map(lambda x: P.token_dict_rev[x],
            [x for (j, x) in enumerate(predicts[0]) if j - 1 in redactions])
        )
    # predictions_hardcoded = list(map(lambda x: token_dict_rev[x], predicts[0][3:5]))
    print(f'Predictions: {predictions_parameterized}')

    # print(f'Hardcoded fill with: {predictions_hardcoded}')
    # list(map(lambda x: token_dict_rev[x], predicts[0][1:3]))
    print(f'.    Actual: {[t for (i, t) in enumerate(tokens_original) if i - 1 in redactions]}')
    print()
    print()
    # if len(predictions) > 10:
    #     break
    return (predictions_parameterized, text)


if __name__ == '__main__':
    df = clean_dataframe()
    prefix_tokens, suffix_tokens = find_first_hom_tokens(df)
    print(f'prefix_tokens: {prefix_tokens}\nsuffix_tokens: {suffix_tokens}')
    print(unredact_tokens(prefix_tokens=prefix_tokens, suffix_tokens=suffix_tokens, num_redactions=5))

    predictions = []
    for sentnum, text in enumerate(sentences):
        print(sentnum)
        predictions.append(unredact_text(text))

# sentence_1 = text
# sentence_2 = 'Joseph Conrad said "We live as we dream, alone." '
# print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))
# indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2, max_len=512)
# masks = np.array([[0] * 512])

# predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]
# print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))

# sentence_2 = 'Mathematicians use patterns to formulate new conjectures; they resolve the truth or falsity of conjectures with proof. '
# print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))
# indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2, max_len=512)

# predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]
# print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))
