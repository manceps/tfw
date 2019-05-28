"""
    A robot Bernie Sanders
    Requirements:
        pip install keras
"""
import numpy as np
import random
import sys
import traceback
import re
import train_bernie
from train_bernie import build_model, predict, make_char_lookup_table

MODEL_FILENAME = 'models/bern.iter999.h5'
#MODEL_FILENAME = 'wiki.iter999.h5'
TEXT_FILENAME = 'bernie_corpus.txt'
#TEXT_FILENAME = '../wiki_corpus.txt'


def main():
    text = train_bernie.read_text_from_file(TEXT_FILENAME)
    char_indices, indices_char = make_char_lookup_table(text)
    model = load_model(char_indices)
    ask_bernie(model, '', char_indices, indices_char)
    while True:
        try:
            question = input('> ')
            ask_bernie(model, question, char_indices, indices_char)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            print('Error, Entering debugger')
            import pdb; pdb.set_trace()


def load_model(char_indices):
    char_count = len(char_indices)
    model = build_model(char_count, batch_size=1)

    print('Decompressing weights...')
    print('Loading weights into model...')
    model.load_weights(MODEL_FILENAME)
    print('Finished loading weights.')
    return model


def ask_bernie(model, question, char_indices, indices_char):
    sentence = re.sub(r'\W+', ' ', question).lower()

    sys.stdout.write("Output: ")
    c = '.'
    for c in sentence:
        predict(model, c, char_indices, indices_char, temperature=0.0, batch_size=1)

    characters = []
    for i in range(512):
        temp = 0.8 if c == ' ' else 0.2
        c = predict(model, c, char_indices, indices_char, temperature=0.1, batch_size=1)
        sys.stdout.write(c)
        sys.stdout.flush()
        characters.append(c)
    sys.stdout.write('\n')
    model.reset_states()

    generated = ''.join(characters)
    # Strip off last word and capitalize sentences
    generated = ' '.join(generated.split()[:-1])
    generated = '. '.join([s.strip().capitalize() for s in generated.split('.')]) + '.'
    return generated


def ask_mueller(model, question, char_indices, indices_char):
    sentence = re.sub(r'\W+', ' ', question).lower()

    sys.stdout.write("Output: ")
    c = '.'
    for c in sentence:
        predict(model, c, char_indices, indices_char, temperature=0.0, batch_size=1)

    characters = []
    for i in range(512):
        temp = 0.8 if c == ' ' else 0.2
        c = predict(model, c, char_indices, indices_char, temperature=0.1, batch_size=1)
        sys.stdout.write(c)
        sys.stdout.flush()
        characters.append(c)
    sys.stdout.write('\n')
    model.reset_states()

    generated = ''.join(characters)
    # Strip off last word and capitalize sentences
    generated = ' '.join(generated.split()[:-1])
    generated = '. '.join([s.strip().capitalize() for s in generated.split('.')]) + '.'
    return generated


if __name__ == '__main__':
    main()
