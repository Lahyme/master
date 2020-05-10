from collections import defaultdict



def read_data(path_1, path_2, path_3):

    with open(path_1, 'r', encoding='utf-8') as f_1, \
            open(path_2, 'r', encoding='utf-8') as f_2, \
            open(path_3, 'r', encoding='utf-8') as f_3:
        words = []
        for line in f_1:
            words += line.split(' ')

        for line in f_2:
            words += line.split(' ')

        for line in f_3:
            words += line.split(' ')

    return words


def build_vocab(items, sort=True, min_count=0, lower=False):

    result = []
    if sort:
        word_dict = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                word_dict[i] += 1

        word_dict = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
        for i, item in enumerate(word_dict):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:

        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)

    vocab = [(w, i) for i, w in enumerate(result)]
    r_vocab = [(i, w) for i, w in enumerate(result)]

    return vocab, r_vocab


def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for word in vocab:
            w, i = word
            f.write("%s\t%d\n" % (w, i))

if __name__ == '__main__':

    lines = read_data('../data/train_set.seg_x.txt',
                      '../data/train_set.seg_y.txt',
                      '../data/test_set.seg_x.txt')
    vocab, reverse_vocab = build_vocab(lines)
    save_word_dict(vocab, '../data/vocab.txt')
