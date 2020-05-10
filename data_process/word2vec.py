from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd


# from utils.data_utils import dump_pkl


def read_lines(path, col_sep=None):
    lines = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def sum_sentences(train_x_seg_path, train_y_seg_path, test_seg_path):
    ret = []
    lines = read_lines(train_x_seg_path)
    lines += read_lines(train_y_seg_path)
    lines += read_lines(test_seg_path)
    for line in lines:
        ret.append(line)
    return ret


def save_sentence(lines, sentence_path):
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    print('save sentence:%s' % sentence_path)


def build_dict(train_x_seg_path, test_y_seg_path, test_seg_path,  sentence_path='',
               w2v_bin_path="w2v.bin", min_count=100):
    sentence = sum_sentences(train_x_seg_path, test_y_seg_path, test_seg_path)
    save_sentence(sentence, sentence_path)

    print('训练模型中')

    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path),
                   size=256, window=5, min_count=min_count, iter=5)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)

    sim = w2v.wv.similarity('技师', '车主')
    print('技师 vs 车主 similarity score:', sim)

    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)

    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    return word_dict


def save_word_dict(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in data.items():  # 遍历字典中的键值
            s2 = str(v)  # 把字典的值转换成字符型
            f.write((k + s2) + '\n')

            # dump_pkl(word_dict, out_path, overwrite=True)


def build_embedding_matrix(vocab, word_path):
    # embeddings_index = {}
    words_index = []
    # embedding_matrix = np.zeros((len(words_index) + 1, 256))
    embeddings_index = {}
    embedding_matrix = {}
    for w, v in vocab.items():
        coefs = np.asarray(v, dtype='float32')
        embeddings_index[w] = coefs
    with open(word_path, 'r', encoding='utf-8') as f_1:
        for line in f_1:
            words_index += line.split(' ')

        for word in words_index:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[word] = embedding_vector

    return embedding_matrix


if __name__ == '__main__':
    word_d = build_dict('data/train_set.seg_x.txt',
                        'data/train_set.seg_y.txt',
                        'data/test_set.seg_x.txt',
                        # out_path='../data/word2vec.txt',
                        sentence_path='data/sentences.txt', )
    save_word_dict('data/word_dict.txt', word_d)
    train_x = build_embedding_matrix(word_d, 'data/train_set.seg_x.txt')
    train_y = build_embedding_matrix(word_d, 'data/train_set.seg_y.txt')
    test_x = build_embedding_matrix(word_d, 'data/test_set.seg_x.txt')