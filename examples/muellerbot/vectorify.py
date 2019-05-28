"""
Usage:
        vectorify.py --dictionary <glove.txt> --input <corpus.txt> --tag <name> [--model <filename.h5>]

Arguments:
        -d, --dictionary <glove.txt>    A pretrained word vector model
        -i, --input <corpus.txt>        Input text to train on
        -t, --tag <name>                Name to identify this experiment
        -m, --model <filename.h5>       Saved parameters to load
"""
import re
import sys
import docopt
import numpy as np
import random
import time


def parse_dictionary(lines, scale_factor=.2):
    word_vectors = {}
    i = 0
    sys.stdout.write("\n")
    for line in lines:
        tokens = line.strip().split()
        if len(tokens) < 2:
            print(("Warning, error parsing word vector on line: {}".format(line)))
            continue
        word = tokens[0]
        vector = np.asarray([[float(n) for n in tokens[1:]]])[0]
        word_vectors[word] = vector * scale_factor
        if i % 1000 == 0:
            sys.stdout.write("\rProcessed {}/{} ({:.01f} percent)    ".format(i, len(lines), 100.0 * i / len(lines)))
            sys.stdout.flush()
        i += 1
    sys.stdout.write("\n")

    print(("Finished parsing {} words".format(len(word_vectors))))
    return word_vectors


def corpus_vocabulary(words):
    word_to_idx = {}
    idx_to_word = {}
    for idx, word in enumerate(set(words)):
        word_to_idx[word] = idx 
        idx_to_word[idx] = word
    return word_to_idx, idx_to_word


def closest_word(word2vec, unknown_vector):
    best_distance = 1000
    best_word = '?'
    best_vector = unknown_vector
    for word, vector in list(word2vec.items()):
        distance = np.linalg.norm(unknown_vector - vector)
        if distance < best_distance:
            best_distance = distance
            best_word = word
            best_vector = vector
    return best_word, best_vector


def build_model(wordvectors, batch_size, vocabulary):
    """
    For a 100-dimensional GloVe word vector mapping, input to the network is a set
    of batch_size vectors, each of length 100.
    Output is softmax among all possible output words in the vocabulary
    """
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Dropout
    from keras.layers.recurrent import GRU
    from keras.optimizers import RMSprop
    word_count, dimensionality = wordvectors.shape
    print('Compiling model...')
    model = Sequential()
    model.add(GRU(512, return_sequences=True, batch_input_shape=(batch_size, 1, dimensionality), stateful=True))
    model.add(Dropout(0.2))
    model.add(GRU(512, return_sequences=False, stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(len(vocabulary)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    print('Finished compiling model')
    return model


def generate_training_data(wordvectors, words, batch_size, word_to_idx):
    assert len(wordvectors) == len(words)
    word_count, dimensionality = wordvectors.shape
    output_dim = len(word_to_idx)
    indices = [random.randint(1, len(wordvectors) - 1) for i in range(batch_size)]
    while True:
        X = np.array([wordvectors[i] for i in indices]).reshape( (batch_size, 1, dimensionality) )
        y = np.zeros((batch_size, len(word_to_idx)))
        for idx, row in enumerate(y):
            word_idx = word_to_idx[words[i]]
            row[word_idx] = 1
        yield (X, y)
        for i in range(len(indices)):
            indices[i] += 1
            if indices[i] >= len(wordvectors) - 1:
                indices[i] = 0


def main(dict_file, corpus_file, tag, model_filename=None):
    print(("Loading dictionary {}".format(dict_file)))
    lines = open(dict_file).readlines()
    print(("Loaded {} lines".format(len(lines))))
    word2vec = parse_dictionary(lines)

    print(("Converting training data {} to word vectors...".format(corpus_file)))
    text = open(corpus_file).read()
    text = re.sub(r'[^a-zA-Z0-9\.]+', ' ', text).lower().replace('.', ' . ')
    words = text.split()
    wordvectors = np.asarray([word2vec.get(word) for word in words if word in word2vec])
    words = np.asarray([word for word in words if word in word2vec])
    print(("Vectorized {} words".format(len(words))))
    word_count, dimensionality = wordvectors.shape

    word_to_idx, idx_to_word = corpus_vocabulary(words)
    print(("Found {} distinct words in corpus".format(len(word_to_idx))))

    batch_size = 128
    model = build_model(wordvectors, batch_size, word_to_idx)
    if model_filename:
        print(("Loading weights from file {}".format(model_filename)))
        model.load_weights(model_filename)
    generator = generate_training_data(wordvectors, words, batch_size, word_to_idx)

    start_time = time.time()
    for iteration in range(1000):
        print(("Starting iteration {} after {} seconds".format(iteration, time.time() - start_time)))
        model.reset_states()
        batches_per_minute = 2 ** 18 / batch_size 
        for i in range(batches_per_minute):
            X, y = next(generator)
            results = model.train_on_batch(X, y)
            sys.stdout.write("\rBatch {} Loss: {}\t".format(i, results))
            sys.stdout.flush()
        sys.stdout.write('\n')

        input_len = 20
        idx = random.randint(0, wordvectors.shape[0] - input_len)
        context_vectors = wordvectors[idx:idx + input_len]
        context_words = words[idx:idx + input_len]

        model.reset_states()
        print("Input: ")
        for vector, word in zip(context_vectors, context_words):
            in_array = np.zeros( (batch_size, 1, dimensionality) )
            in_array[0] = vector
            sys.stdout.write(' ' + word)
            model.predict(in_array, batch_size=batch_size)[0]
        sys.stdout.write('\n')

        print("Output: ")
        predicted_word = word
        in_array = np.zeros( (batch_size, 1, dimensionality) )
        for i in range(5):
            in_array[0] = word2vec[predicted_word]
            y = model.predict(in_array, batch_size=batch_size)[0]
            predicted_word = idx_to_word[y.argmax()]
            sys.stdout.write(" {}".format(predicted_word))
            sys.stdout.flush()
        sys.stdout.write('\n')

        model.save_weights('models/{}.iter{}.h5'.format(tag, iteration), overwrite=True)
    print("Finished")


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    dict_file = arguments['--dictionary']
    corpus_file = arguments['--input']
    tag = arguments['--tag']
    model = arguments['--model']
    main(dict_file, corpus_file, tag, model)
