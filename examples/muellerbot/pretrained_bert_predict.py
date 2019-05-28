import sys
import os
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint, Tokenizer


REDACTION_TOKEN = '[MASK]'

UNZIPPED_MODEL_PATH = None
if len(sys.argv) == 2:
    UNZIPPED_MODEL_PATH = os.path.abspath(sys.argv[1])

UNZIPPED_MODEL_PATH = (
    UNZIPPED_MODEL_PATH or os.environ.get('UNZIPPED_MODEL_PATH') or os.path.abspath(
        '~/midata/public/models/bert/multi_cased_L-12_H-768_A-12'))


def load_model():
    if len(sys.argv) != 4:
        print('python load_model.py CONFIG_PATH CHECKPOINT_PATH DICT_PATH')
        print('CONFIG_PATH:     $UNZIPPED_MODEL_PATH/bert_config.json')
        print('CHECKPOINT_PATH: $UNZIPPED_MODEL_PATH/bert_model.ckpt')
        print('DICT_PATH:       $UNZIPPED_MODEL_PATH/vocab.txt')
        sys.argv = [
            sys.argv[0],
            os.environ.get('CONFIG_PATH') or os.path.join(UNZIPPED_MODEL_PATH, 'bert_config.json'),
            os.environ.get('CHECKPOINT_PATH') or os.path.join(UNZIPPED_MODEL_PATH, 'bert_model.ckpt'),
            os.environ.get('DICT_PATH') or os.path.join(UNZIPPED_MODEL_PATH, 'vocab.txt'),
        ]

    if not all([os.path.exists(p) for p in sys.argv[1:4]]):
        print("You must specify the path where you've downloaded the pretrained BERT model in $UNZIPPED_MODEL_PATH or on the commandline.")

    return sys.argv

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


def unredact(
        pipeline=None,
        text='Mathematics is a discipline that uses symbolic language to study concepts such as quantity, structure, change, and space.',
        redacted_token_positions=[1, 2],
        redaction_token=REDACTION_TOKEN):
    pipeline = pipeline or load_model()
    model = pipeline['model']
    tokenizer = pipeline['tokenizer']
    token_dict = pipeline['token_dict']
    token_dict_rev = pipeline['token_dict_rev']

    tokens = tokenizer.tokenize(text)
    for i in redacted_token_positions:
        tokens[i] = redaction_token

    print('Tokens:', tokens)
    indices = np.asarray([[token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
    segments = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens))])
    masks = np.asarray([[0, 1, 1] + [0] * (512 - 3)])

    predicts = model.predict([indices, segments, masks])[0]
    predicts = np.argmax(predicts, axis=-1)
    unredacted_tokens = list(map(lambda x: token_dict_rev[x], [predicts[0][i] for i in redacted_token_positions]))
    print('Fill with: ', unredacted_tokens)


def is_next_random(pipeline=None,
                   sentence_1='Mathematics is uses symbolic language to study concepts such as quantity, structure, and space.',
                   sentence_2='Joseph Conrad wrote "We live as we dream, alone."',
                   ):
    pipeline = pipeline or load_model()
    model = pipeline['model']
    tokenizer = pipeline['tokenizer']
    # token_dict = pipeline['token_dict']
    # token_dict_rev = pipeline['token_dict_rev']

    print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))
    indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2, max_len=512)
    masks = np.array([[0] * 512])

    predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]
    print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))
    return bool(np.argmax(predicts, axis=-1)[0])


def demo_is_next_random(pipeline=None,
                        sentence_1='Mathematics is uses symbolic language to study concepts such as quantity, structure, and space.',
                        ):
    pipeline = pipeline or load_model()

    print()
    sentences = [
        'Joseph Conrad wrote: Colonel Kurtz said, "we live as we dream, alone." ',
        'A neural network is a computational graph trained on the statistics of a dataset to form a mathematical model. ',
        'Mathematicians seek and use patterns to formulate new conjectures; they resolve the truth or falsity of conjectures by mathematical proof. '
    ]

    is_random = []
    for s in sentences:
        is_random = is_next_random(pipeline=pipeline, sentence_1=sentence_1, sentence_2=s)

    return list(zip(sentences, is_random))


def interact(pipeline=None):
    pipeline = pipeline or load_model()

    redacted = ' '
    while len(redacted):
        redacted = input('Redacted text: ')
        unredacted = unredact(pipeline=pipeline, text=redacted)
        print(f'Unredacted text: {unredacted}')
        print()


if __name__ == '__main__':
    interact(pipeline=load_model())
