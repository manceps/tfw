import sys
import os
import codecs
import re
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from find_redactions import clean_dataframe, get_line_pairs, find_repeated_substring


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

global P
P = None


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


# (TEXT, REDACTION_MARKER, PAGE_NUM, NUM_WORDS_IN_REDACTION)
TEXTS = [
    ('''
    The presidential campaign of Donald J. Trump ("Trump Campaign" or "Campaign") showed
    interest in WikiLeaks\'s releases of documents and welcomed their potential to damage
    candidate Clinton. Beginning in June 2016, [Harm to Ongoing Matter] forecast to senior
    Campaign officials that WikiLeaks would release information damaging to candidate Clinton.
    WikiLeaks\'s first release came in July 2016. Around the same time, candidate Trump announced
    that he hoped Russia would recover emails described as missing from a private server used by
    linton when she was Secretary of State (he later said that he was speaking sarcastically).
    ''', '[Harm to Ongoing Matter]', 5, 4),

    ('''
    On October 20, 2017, the Acting Attorney General confirmed in a memorandum the Special
    Counsel's investigative authority as to several individuals and entities. First, as part of
    a full and thorough investigation of the Russian government's efforts to interfere in the
    2016 presidential election," the Special Counsel was authorized to investigate the pertinent
    ctivities of Michael Cohen, Richard Gates, [Personal Privacy] , Roger Stone, and
    ''', '[Personal Privacy]', 12, 2),

    ('''By February 2016, internal IRA documents referred to support for the Trump Campaign
    and opposition to candidate Clinton.49 For example, [Harm to Ongoing Matter] directions to
    IRA operators
    ''', '[Harm to Ongoing Matter]', 23, 5),

    ('''The focus on the U.S. presidential campaign continued throughout 2016.
    In [Harm to Ongoing Matter] 2016 internal [Harm to Ongoing Matter] reviewing the
    IRA-controlled Facebook group "Secured Borders,"
    ''', '[Harm to Ongoing Matter]', 23, 3),

    ('''IRA employees frequently used Investigative Technique Twitter, Facebook, and Instagram
    to contact and recruit U.S. persons who followed the group. The IRA recruited U.S. persons
    from across the political spectrum. For example, the IRA targeted the family
    of [Personal Privacy] and a number of black social justice activists
    ''', '[Personal Privacy]', 31, 3),

    ('''
    A. GRU Hacking Directed at the Clinton Campaign 1. GRU Units Target the Clinton Campaign Two military units of the GRU
    carried out the computer intrusions into the Clinton Campaign, DNC, and DCCC: Military Units 26165 and 74455.110
    Military Unit 26165 is a GRU cyber unit dedicated to targeting military, political, governmental, and non-governmental
    organizations outside of Russia, including in the United States.111 The unit was sub-divided into departments with
    different specialties. One department, for example, developed specialized malicious software ("malware"), while another
    department conducted large-scale spearphishing campaigns. 112 Investigative Technique a bitcoin mining operation
    to 109 As discussed in Section V below, our Office charged 12 GRU officers for crimes arising from the hacking of these
    computers, principally with conspiring to commit computer intrusions, in violation of 18 U.S.C. $$1030 and 371.
    See Volume 1, Section V.B, infra; Indictment, United States v. Netyksho, No., [Investigative Technique] a bitcoin mining
    operation to secure bitcoins used to purchase computer infrastructure used in hacking operations.
    ''', "[Investigative Technique]", 36, 3),

    ('''
    Footnote: 113. Bitcoin mining consists of unlocking new bitcoins by solving computational problems. [IT] kept its
    newly mined coins in an account on the bitcoin exchange platform CEX.io. To make purchases, the GRU routed funds
    into other accounts through transactions designed to obscure the source of funds. Netyksho Indictment 62.
    ''', '[IT]', 37, 2),

    ('''
    The first set of GRIU-controlled computers, known by the GRU as "middle servers," sent and received messages to and
    from malware on the DNC/DCCC networks. The middle servers, in turn, relayed messages to a second set of
    GRU-controlled computers, labeled internally by the GRU as an "AMS Panel." The AMS Panel [Investigative Technique]
    served as a nerve center through which GRU officers monitored and directed the malware's operations on the
    DNC/DCCC networks. 127 The AMS Panel used to control X-Agent during the DCCC and DNC intrusions was housed on a
    leased computer located near IT Arizona.
    ''', '[Investigative Technique]', 39, 3),

    ('''Footnote: 140. See, e.g., Internet Archive, "https://dcleaks.com/" (archive date Nov. 10, 2016). Additionally,
    DCLeaks released documents relating to [Personal Privacy] , emails belonging to [PP] , and emails from  2015 relating
    to Republican Party employees (under the portfolio name "The United States Republican Party\').
    "The United States Republican Party" portfolio contained approximately 300 emails from a variety of GOP members, PACs,
    campaigns, state parties, and businesses dated between May and October 2015. According to open-source reporting,
    these victims shared the same Tennessee-based web-hosting company, called Smartech Corporation.'
    ''', '[Personal Privacy]', 41, 5),

    ('''Footnote: 140. See, e.g., Internet Archive, "https://dcleaks.com/" (archive date Nov. 10, 2016). Additionally,
    DCLeaks released documents relating to [PP], emails belonging to [Personal Privacy] , and emails from  2015 relating
    to Republican Party employees (under the portfolio name "The United States Republican Party\').
    "The United States Republican Party" portfolio contained approximately 300 emails from a variety of GOP members, PACs,
    campaigns, state parties, and businesses dated between May and October 2015. According to open-source reporting,
    these victims shared the same Tennessee-based web-hosting company, called Smartech Corporation.'
    ''', '[Personal Privacy]', 41, 1),
    ]

# to manually generate plausible redaction unredactions:
# df = clean_dataframe()

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


def find_first_hom_tokens(df, text=None, substring='of documents and', marker='[Personal Privacy]'):
    global P
    P = P or load_pipeline()
    if not text:
        df = clean_dataframe(df) if isinstance(df, str) else df
        for t in df.text:
            if substring in t:
                text = t
                break
    # print(f'TEXT: {text}')
    tokens = P.tokenizer.tokenize(text)
    joined_tokens = ' '.join(tokens)
    # print(f'joined_tokens: {joined_tokens}')
    hom = ' '.join(P.tokenizer.tokenize(marker)[1:-1])
    # print(f'joined_hom: {hom}')
    hom_start = joined_tokens.find(hom)
    hom_stop = hom_start + len(hom)
    # print(f'hom_start: {hom_start}, hom_stop: {hom_stop}')
    prefix_tokens = joined_tokens[:hom_start].split()
    suffix_tokens = joined_tokens[hom_stop:].split()
    # print(f'HOM prefix_tokens: {prefix_tokens}\nHOM suffix_tokens: {suffix_tokens}')

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
    global P
    if not P:
        P = load_pipeline()
    tokens = list(prefix_tokens) + [MASK_TOKEN] * num_redactions + list(suffix_tokens)
    tokens = tokens[:512]
    tokens_original = tokens.copy()
    # text = ' '.join(tokens)
    # print(f"Predicting {num_redactions} MASK tokens in: {' '.join(tokens)}")

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
    return (predictions_parameterized, tokens)


def unredact_text(text, redactions=[2, 3]):
    print(f"Redacting tokens {redactions} in: {text}")
    global P
    P = P or load_pipeline()

    tokens = P.tokenizer.tokenize(text)
    tokens_original = tokens.copy()
    for r in redactions:
        tokens[r + 1] = MASK_TOKEN

    # print(f'Tokens: {tokens}')

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
    print(f'Predictions: {" ".join(predictions_parameterized)}')

    # print(f'Hardcoded fill with: {predictions_hardcoded}')
    # list(map(lambda x: token_dict_rev[x], predicts[0][1:3]))
    print(f'.    Actual: {[t for (i, t) in enumerate(tokens_original) if i - 1 in redactions]}')
    print()
    print()
    # if len(predictions) > 10:
    #     break
    return (predictions_parameterized, text)


def unredact_examples(examples=TEXTS):
    for text, marker, page, num_redactions in examples:
        print('\n\n****************************')
        prefix_tokens, suffix_tokens = find_first_hom_tokens(df=None, text=text, marker=marker)
        print(f'page: {page}\nnum_words: {num_redactions}\nprefix_tokens: {prefix_tokens}\nsuffix_tokens: {suffix_tokens}')
        print(unredact_tokens(prefix_tokens=prefix_tokens, suffix_tokens=suffix_tokens, num_redactions=num_redactions))
        print('\n\n****************************')

    predictions = []
    for sentnum, text in enumerate(sentences):
        print(sentnum)
        predictions.append(unredact_text(text))


def unredact_interactively():
    global P
    if not P:
        P = load_pipeline()
    unredacted = ' '
    while unredacted:
        text = input('Text: ')
        marker = input('Redaction marker: ')
        marker = marker or 'unk'
        redactions = find_repeated_substring(text, substring=marker)
        if not redactions:
            print('No redactions found')
            unredacted = text
            continue
        # print(redactions)
        start, stop = redactions[0], redactions[-1] + len(marker)
        prefix, suffix = text[:start], text[stop:]
        # print(start, stop)
        # print(f'prefix: {prefix}')
        # print(f'suffix: {suffix}')
        prefix_tokens = P.tokenizer.tokenize(prefix)[:-1]
        suffix_tokens = P.tokenizer.tokenize(suffix)[1:]
        # print(f'prefix_tokens: {prefix_tokens}')
        # print(f'suffix_tokens: {suffix_tokens}')
        unredacted_tokens, all_tokens = unredact_tokens(prefix_tokens=prefix_tokens, suffix_tokens=suffix_tokens, num_redactions=len(redactions))
        print(f'all_tokens: {all_tokens}')
        print(f'unredacted_tokens: {unredacted_tokens}')
        j = 0
        for (i, tok) in enumerate(all_tokens):
            if tok == '[MASK]' and j < len(unredacted_tokens):
                all_tokens[i] = unredacted_tokens[j]
                j += 1

        unredacted = ' '.join(all_tokens)
        # unredacted = ' '.join([t[2:] if t.startswith('##') else t for t in unredacted_tokens])
        print(f'Unredacted text: {unredacted}')


if __name__ == '__main__':
    unredact_interactively()

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
