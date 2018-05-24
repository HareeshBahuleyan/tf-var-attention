import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

import sys

if not '../' in sys.path: sys.path.append('../')

import pandas as pd

from utils import data_utils
from model_config import config
from ded_detAttn import DetSeq2SeqDetAttnModel


def train_model(config):
    print('[INFO] Preparing data for experiment: {}'.format(config['experiment']))
    if config['experiment'] == 'qgen':
        train_data = pd.read_csv(config['data_dir'] + 'df_qgen_train.csv')
        val_data = pd.read_csv(config['data_dir'] + 'df_qgen_val.csv')
        test_data = pd.read_csv(config['data_dir'] + 'df_qgen_test.csv')
        input_sentences = pd.concat([train_data['answer'], val_data['answer'], test_data['answer']])
        output_sentences = pd.concat([train_data['question'], val_data['question'], test_data['question']])
        true_val = val_data['question']
        filters = '!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n'
        w2v_path = config['w2v_dir'] + 'w2vmodel_qgen.pkl'

    elif config['experiment'] == 'dialogue':
        train_data = pd.read_csv(config['data_dir'] + 'df_dialogue_train.csv')
        val_data = pd.read_csv(config['data_dir'] + 'df_dialogue_val.csv')
        test_data = pd.read_csv(config['data_dir'] + 'df_dialogue_test.csv')
        input_sentences = pd.concat([train_data['line'], val_data['line'], test_data['line']])
        output_sentences = pd.concat([train_data['reply'], val_data['reply'], test_data['reply']])
        true_val = val_data['reply']
        filters = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
        w2v_path = config['w2v_dir'] + 'w2vmodel_dialogue.pkl'

    else:
        print('Invalid experiment name specified!')
        return

    print('[INFO] Tokenizing input and output sequences')
    x, input_word_index = data_utils.tokenize_sequence(input_sentences,
                                                       filters,
                                                       config['encoder_num_tokens'],
                                                       config['encoder_vocab'])

    y, output_word_index = data_utils.tokenize_sequence(output_sentences,
                                                        filters,
                                                        config['decoder_num_tokens'],
                                                        config['decoder_vocab'])

    print('[INFO] Split data into train-validation-test sets')
    x_train, y_train, x_val, y_val, x_test, y_test = data_utils.create_data_split(x,
                                                                                  y,
                                                                                  config['experiment'])

    encoder_embeddings_matrix = data_utils.create_embedding_matrix(input_word_index,
                                                                   config['embedding_size'],
                                                                   w2v_path)

    decoder_embeddings_matrix = data_utils.create_embedding_matrix(output_word_index,
                                                                   config['embedding_size'],
                                                                   w2v_path)

    # Re-calculate the vocab size based on the word_idx dictionary
    config['encoder_vocab'] = len(input_word_index)
    config['decoder_vocab'] = len(output_word_index)

    model = DetSeq2SeqDetAttnModel(config,
                                   encoder_embeddings_matrix,
                                   decoder_embeddings_matrix,
                                   input_word_index,
                                   output_word_index)

    model.train(x_train, y_train, x_val, y_val, true_val)


if __name__ == '__main__':
    train_model(config)
