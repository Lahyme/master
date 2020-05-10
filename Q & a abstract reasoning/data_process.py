import numpy as np
import pandas as pd
import re
from jieba import posseg
import jieba


REMOVE_WORDS = ['|', '[', ']', '语音', '图片']

def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line.strip()
            lines.add(line)

    return lines


def remove_stopwords(words_list):
    words_list = [word for word in words_list if word not in REMOVE_WORDS]
    return words_list

def segment(sentence, cut_type = 'word', pos = False):
    if pos:
        if cut_type == 'word':
            word_posseq = posseg.lcut(sentence)
            word_seq = []
            pos_seq = []
            for w, p in word_posseq:
                word_seq.append(w)
                pos_seq.append(p)
        elif cut_type =='char':
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
        return word_seq, pos_seq
    else:
        if cut_type == 'word':
            word_seq = jieba.lcut(sentence)
        elif cut_type == 'char':
            word_seq = list(sentence)
        return word_seq



def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, encoding='utf-8')
    train_df.dropna(subset=['Report'], how='any', inplace=True)
    train_df.fillna('缺失', inplace=True)
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    train_y = []
    if 'Report' in train_df.columns:
        train_y = train_df.Report
        assert len(train_x) == len(train_y)

    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.fillna('缺失', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_y = []

    return train_x, train_y, test_x, test_y


def save_data(train_x, train_y, test_y, train_x_path, train_y_path, test_y_path, stop_words_path):
    stop_words = read_stopwords(stop_words_path)

    with open(train_x_path, mode='w', encoding='utf-8') as f_1:
        len_1 = 0
        for line in train_x:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                seg_list = remove_stopwords(seg_list)
                seg_list = [word for word in seg_list if word not in stop_words]
                if len(seg_list) > 0:
                    sen = ' '.join(seg_list)
                    f_1.write('%s' % sen)
                    f_1.write('\n')
                    len_1 += 1
        print('train_x的长度是', len_1)

    with open(train_y_path, mode='w', encoding='utf-8') as f_2:
        len_2 = 0
        for line in train_y:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                #seg_list = remove_stopwords(seg_list)
                seg_list = [word for word in seg_list if word not in stop_words]
                if len(seg_list) > 0:
                    sen = ' '.join(seg_list)
                    f_2.write('%s' % sen)
                    f_2.write('\n')
                    len_2 += 1
                else:
                    f_2.write("随时 联系")
                    f_2.write('\n')
                    # print('缺失')
                    len_2 += 1

        print('train_y的长度是', len_2)

    with open(test_y_path, mode='w', encoding='utf-8') as f_3:
        len_3 = 0
        for line in test_y:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                seg_list = remove_stopwords(seg_list)
                seg_list = [word for word in seg_list if word not in stop_words]
                if len(seg_list) > 0:
                    sen = ' '.join(seg_list)
                    f_3.write('%s' % sen)
                    f_3.write('\n')
                    len_3 += 1

        print('test_y的长度是', len_3)


def preprocess (sentence):
    seg_list = segment(sentence.strip(), cut_type='word')
    sen = ' '.join(seg_list)
    return sen


if __name__ == '__main__':
    #还没写完
    train_x_list, train_y_list, test_x_list, _ = load_data('../data/AutoMaster_TrainSet.csv',
                                                                  '../data/AutoMaster_TestSet.csv')

    print(len(train_x_list))
    print(len(train_y_list))

    save_data(train_x_list,
              train_x_list,
              test_x_list,
              '../data/train_set.seg_x.txt',
              '../data/train_set.seg_y.txt',
              '../data/test_set.seg_x.txt',
              stop_words_path='../data/stop_words.txt')
