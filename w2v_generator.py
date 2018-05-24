import os
import argparse
import gensim
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize


W2V_DIR = 'w2v_models/'
DATA_DIR = 'data/'

parser = argparse.ArgumentParser(description='Create word2vec embeddings for the specified dataset')
parser.add_argument('-d', '--dataset', help='Specify dataset: either qgen, dialogue or both', required=True)
args = vars(parser.parse_args())


def main():
    if not os.path.exists(W2V_DIR):
        os.mkdir(W2V_DIR)

    all_files = os.listdir(DATA_DIR)

    if args['dataset'] == 'dialogue':
        files = [f for f in all_files if 'dialogue' in f]
    elif args['dataset'] == 'qgen':
        files = [f for f in all_files if 'qgen' in f]
    else:
        print('Invalid Argument !')
        return

    df_list = pd.concat(load_data(files))
    df_list.reset_index(inplace=True, drop=True)
    data = list(df_list.iloc[:, 0] + df_list.iloc[:, 1]) # 1st and 2nd column
    create_w2v(data)
    print('Word2Vec created successfully for {}'.format(args['dataset']))


def load_data(files):
    df_list = []
    for f in files:
        df_list.append(pd.read_csv(DATA_DIR + f))

    return df_list


def create_w2v(sentences):
    np.random.shuffle(sentences)
    sentences = [word_tokenize(s) for s in sentences]
    w2v_model = gensim.models.Word2Vec(sentences,
                                       size=300,
                                       min_count=1,
                                       iter=50)
    w2v_model.save(W2V_DIR + 'w2vmodel_' + args['dataset'] + '.pkl')


if __name__ == '__main__':
    main()
