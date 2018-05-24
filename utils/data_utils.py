import numpy as np
import pandas as pd
import gensim
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def tokenize_sequence(sentences, filters, max_num_words, max_vocab_size):
    """
    Tokenizes a given input sequence of words.

    Args:
        sentences: List of sentences
        filters: List of filters/punctuations to omit (for Keras tokenizer)
        max_num_words: Number of words to be considered in the fixed length sequence
        max_vocab_size: Number of most frequently occurring words to be kept in the vocabulary

    Returns:
        x : List of padded/truncated indices created from list of sentences
        word_index: dictionary storing the word-to-index correspondence

    """

    sentences = [' '.join(word_tokenize(s)[:max_num_words]) for s in sentences]

    tokenizer = Tokenizer(filters=filters)
    tokenizer.fit_on_texts(sentences)

    word_index = dict()
    word_index['PAD'] = 0
    word_index['UNK'] = 1
    word_index['GO'] = 2
    word_index['EOS'] = 3

    for i, word in enumerate(dict(tokenizer.word_index).keys()):
        word_index[word] = i + 4

    tokenizer.word_index = word_index
    x = tokenizer.texts_to_sequences(list(sentences))

    for i, seq in enumerate(x):
        if any(t >= max_vocab_size for t in seq):
            seq = [t if t < max_vocab_size else word_index['UNK'] for t in seq]
        seq.append(word_index['EOS'])
        x[i] = seq

    x = pad_sequences(x, padding='post', truncating='post', maxlen=max_num_words, value=word_index['PAD'])

    word_index = {k: v for k, v in word_index.items() if v < max_vocab_size}

    return x, word_index


def create_embedding_matrix(word_index, embedding_dim, w2v_path):
    """
    Create the initial embedding matrix for TF Graph.

    Args:
        word_index: dictionary storing the word-to-index correspondence
        embedding_dim: word2vec dimension
        w2v_path: file path to the w2v pickle file

    Returns:
        embeddings_matrix : numpy 2d-array with word vectors

    """
    w2v_model = gensim.models.Word2Vec.load(w2v_path)
    embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(word_index), embedding_dim))
    for word, i in word_index.items():
        try:
            embeddings_vector = w2v_model[word]
            embeddings_matrix[i] = embeddings_vector
        except KeyError:
            pass

    return embeddings_matrix


def create_data_split(x, y, experiment):
    """
    Create test-train split according to previously defined CSV files
    Depending on the experiment - qgen or dialogue

    Args:
        x: input sequence of indices
        y: output sequence of indices
        experiment: dialogue (conversation system) or qgen (question generation) task

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test: train val test split arrays

    """

    if experiment == 'qgen':
        train_size = pd.read_csv('../data/df_qgen_train.csv').shape[0]
        val_size = pd.read_csv('../data/df_qgen_val.csv').shape[0]
        test_size = pd.read_csv('../data/df_qgen_test.csv').shape[0]
    elif experiment == 'dialogue':
        train_size = pd.read_csv('../data/df_dialogue_train.csv').shape[0]
        val_size = pd.read_csv('../data/df_dialogue_val.csv').shape[0]
        test_size = pd.read_csv('../data/df_dialogue_test.csv').shape[0]
    else:
        print('Invalid experiment name specified !')
        return

    train_indices = range(train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, train_size + val_size + test_size)

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_val = x[val_indices]
    y_val = y[val_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_batches(x, y, batch_size):
    """
    Generate inputs and targets in a batch-wise fashion for feed-dict

    Args:
        x: entire source sequence array
        y: entire output sequence array
        batch_size: batch size

    Returns:
        x_batch, y_batch, source_sentence_length, target_sentence_length

    """

    for batch_i in range(0, len(x) // batch_size):
        start_i = batch_i * batch_size
        x_batch = x[start_i:start_i + batch_size]
        y_batch = y[start_i:start_i + batch_size]

        source_sentence_length = [np.count_nonzero(seq) for seq in x_batch]
        target_sentence_length = [np.count_nonzero(seq) for seq in y_batch]

        yield x_batch, y_batch, source_sentence_length, target_sentence_length
