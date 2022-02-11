#!/user/bin/env python3
# -*- coding: utf-8 -*-
import random
from tqdm import tqdm


def split(path: str,
          save_path: str,
          seed=0,
          split_ratio=0.1,
          split_dev=True):
    train_path = save_path+'/train.txt'
    dev_path = save_path+'/dev.txt'
    test_path = save_path + '/test.txt'

    all_data = []
    dev_data = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            all_data.append(line)

    random.seed(seed)
    random.shuffle(all_data)

    length = len(all_data)
    if split_dev:
        train_len = length*(1 - split_ratio*2)
        dev_len = train_len + length*(split_ratio*2*0.25)
        train_len = 18000
        dev_len = 18300
        train_data = all_data[:train_len]
        dev_data = all_data[train_len:dev_len]
        test_data = all_data[dev_len:]
    else:
        train_len = length * (1 - split_ratio)
        train_data = all_data[:train_len]
        test_data = all_data[train_len:]

    with open(train_path, mode='w', encoding='utf-8') as f:
        for i in tqdm(range(len(train_data))):
            f.write(train_data[i])
            f.write('\n')

    with open(test_path, mode='w', encoding='utf-8') as f:
        for i in tqdm(range(len(test_data))):
            f.write(test_data[i])
            f.write('\n')
    if dev_data:
        with open(dev_path, mode='w', encoding='utf-8') as f:
            for i in tqdm(range(len(dev_data))):
                f.write(dev_data[i])
                f.write('\n')

    print("Data have been saved in : {}".format(save_path))


def get_class_mapping(super_mode: str):
    """
    Mapping mode into integer
    :param super_mode: before, after & both
    :return:
    """
    class_mapping = ['none', 'replace']
    if super_mode == 'both':
        class_mapping.extend(['before', 'after'])
    else:
        class_mapping.append(super_mode)
    return {k: v for v, k in enumerate(class_mapping)}
