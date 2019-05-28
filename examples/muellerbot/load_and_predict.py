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

UNZIPPED_MODEL_PATH = None
if len(sys.argv) == 2:
    UNZIPPED_MODEL_PATH = os.path.abspath(os.path.expanduser(sys.argv[1]))
print(UNZIPPED_MODEL_PATH)

UNZIPPED_MODEL_PATH = (
    UNZIPPED_MODEL_PATH or os.environ.get('UNZIPPED_MODEL_PATH') or os.path.abspath(
        os.path.expanduser('~/midata/public/models/bert/uncased_L-12_H-768_A-12')))

print(UNZIPPED_MODEL_PATH)

df = clean_dataframe()
line_pairs = get_line_pairs(df, min_line_length=60, max_line_length=150)
print(pd.DataFrame(line_pairs, columns='text redacted_text'.split()).head())

for i, (text, redacted) in enumerate(line_pairs):
    if len(re.findall(r'^[-:.0-9 \t]{1,2}', text.strip())) > 0:
        print(f'Skipping: {text[:30]}')
        continue
    print(f"Redacting: {text[:30]}")
    print()


if len(sys.argv) != 4:
    print('USAGE: python load_model.py CONFIG_PATH CHECKPOINT_PATH DICT_PATH')
    print()
    sys.argv = [
        sys.argv[0],
        os.environ.get('CONFIG_PATH') or os.path.join(UNZIPPED_MODEL_PATH, 'bert_config.json'),
        os.environ.get('CHECKPOINT_PATH') or os.path.join(UNZIPPED_MODEL_PATH, 'bert_model.ckpt'),
        os.environ.get('DICT_PATH') or os.path.join(UNZIPPED_MODEL_PATH, 'vocab.txt')]
    SCRIPTNAME, CONFIG_PATH, CHECKPOINT_PATH, DICT_PATH = sys.argv
    print(f'CONFIG_PATH:     {CONFIG_PATH}')
    print(f'CHECKPOINT_PATH: {CHECKPOINT_PATH}')
    print(f'DICT_PATH:       {DICT_PATH}')

config_path, checkpoint_path, dict_path = tuple(sys.argv[1:])

model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
model.summary(line_length=120)

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
token_dict_rev = {v: k for k, v in token_dict.items()}

tokenizer = Tokenizer(token_dict)

# original keras-bert demo example in chinese translated to English:
# text = '数学是利用符号语言研究數量、结构、变化以及空间等概念的一門学科。'
# text = 'Google is dedicated to proactive openness and applying machine learning technology to further the common good.'

# original keras-bert demo example in chinese translated to English:
# text = text or 'Mathematics is a discipline that uses symbolic language to study concepts such as quantity, structure, change, and space.'


def load_pipeline(UNZIPPED_MODEL_PATH=UNZIPPED_MODEL_PATH):
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
    model.summary(line_length=140)

    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    token_dict_rev = {v: k for k, v in token_dict.items()}
    tokenizer = Tokenizer(token_dict)

    return dict(model=model, token_dict=token_dict, token_dict_rev=token_dict_rev, tokenizer=tokenizer)


sentences = [
    'The IRA later used social media accounts and interest groups to sow discord in the U.S. political system through what it termed "information warfare."',
    ('The campaign evolved from a generalized program designed in 2014 and 2015 to undermine the U.S. electoral system,' +
        'to a targeted operation that by early 2016 favored candidate Trump and disparaged candidate Clinton.'),
    ('The IRA\'s operation also included the purchase of political advertisements on social media in the names of U.S. persons and entities,' +
        'as well as the staging of political rallies inside the United States.'),
    ('To organize those rallies, IRA employees posed as U.S. grassroots entities and persons and made contact' +
        ' with Trump supporters and Trump Campaign officials in the United States.'),
    ]


predictions = []
redactions = [2, 3]
for i, text in enumerate(sentences):
    print(f"Redacting words {redactions}:")
    tokens = tokenizer.tokenize(text)
    for r in redactions:
        tokens[r + 1] = '[MASK]'
    print(f'Tokens: {tokens}')

    indices = np.asarray([[token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
    segments = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens))])
    masks = np.asarray([[0] * 512])
    for r in redactions:
        masks[0][r + 1] = 1
    # masks = np.asarray([[0, 1, 1] + [0] * (512 - 3)])

    predicts = model.predict([indices, segments, masks])[0]
    predicts = np.argmax(predicts, axis=-1)
    predictions_parameterized = list(map(lambda x: token_dict_rev[x],
        [x for (j, x) in enumerate(predicts[0]) if j + 1 in redactions]))
    predictions.append((predictions_parameterized, text))
    predictions_hardcoded = list(map(lambda x: token_dict_rev[x], predicts[0][3:5]))
    print('Fill with: ', predcitions_hardcoded)

    print(f'New fill with: {predictions[-1][0]}')
    # list(map(lambda x: token_dict_rev[x], predicts[0][1:3]))
    print(f'Actual tokens: {predictions[-1][1][:80]}')
    if len(predictions) > 10:
        break

for i, (text, redacted) in enumerate(line_pairs):
    # try to filter out footnotes, etc
    if re.findall(r'^[-:.0-9 \t]{1,2}', text):
        continue
    pass

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
