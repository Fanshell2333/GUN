#!/user/bin/env python3
# -*- coding: utf-8 -*-
import random
from tqdm import tqdm
from typing import List
from simplediff import diff


class SpecialSymbol:
    cls = '[CLS]'
    context_internal = '[SEP]'
    end_placeholder = '[END]'


def export_word_edit_matrix(context: List,
                            current_sen: List,
                            label_sen: List,
                            super_mode: str = 'before',
                            # if there requires multiple insert, we only
                            # keep the longest one
                            only_one_insert: bool = False):
    if isinstance(context, str):
        context_seq = list(context)
        current_seq = list(current_sen)
        label_seq = list(label_sen)
    else:
        context_seq = context
        current_seq = current_sen
        label_seq = label_sen
    applied_changes = diff(current_seq, label_seq)

    def sub_finder(cus_list, pattern, used_pos):
        find_indices = []
        for i in range(len(cus_list)):
            if cus_list[i] == pattern[0] and \
                    cus_list[i:i + len(pattern)] == pattern \
                    and i not in used_pos:
                find_indices.append((i, i + len(pattern)))
        if len(find_indices) == 0:
            return 0, 0
        else:
            return find_indices[-1]

    def cont_sub_finder(cus_list, pattern, used_pos):
        context_len = len(cus_list)
        pattern_len = len(pattern)
        for i in range(context_len):
            k = i
            j = 0
            temp_indices = []
            while j < pattern_len and k < context_len:
                if cus_list[k] == pattern[j][0] and \
                        cus_list[k:k + len(pattern[j])] == pattern[j] \
                        and k not in used_pos:
                    temp_indices.append((k, k + len(pattern[j])))
                    j += 1
                else:
                    k += 1
            if j == pattern_len:
                return zip(*temp_indices)
        else:
            return 0, 0

    rm_range = None
    ret_ops = []
    context_used_pos = []
    current_used_pos = []
    pointer = 0
    for diff_sample in applied_changes:
        diff_op = diff_sample[0]
        diff_content = diff_sample[1]
        if diff_op == '-':
            if rm_range is not None:
                ret_ops.append(['remove', rm_range, []])
            start, end = sub_finder(current_seq, diff_content, current_used_pos
                                    )
            rm_range = [start, end]
            current_used_pos.extend(list(range(start, end)))
        elif diff_op == '+':
            start, end = sub_finder(context_seq, diff_content, context_used_pos)
            # cannot find the exact match substring, we should identify the snippets
            if start == 0 and end == 0:
                inner_diff = diff(diff_content, context_seq)
                overlap_content = [inner_diff_sample[1] for
                                   inner_diff_sample in inner_diff if inner_diff_sample[0] == '=']
                if len(overlap_content) > 0:
                    # only take one insert
                    if len(overlap_content) == 1 or only_one_insert:
                        overlap_content = sorted(overlap_content, key=lambda x: len(x), reverse=True)[0]
                        start, end = sub_finder(context_seq, overlap_content,
                                                context_used_pos)
                    else:
                        start_end_tuple = cont_sub_finder(context_seq, overlap_content, context_used_pos)
                        # start is a list, end is also
                        start, end = start_end_tuple
                else:
                    start, end = 0, 0
            if not (start == 0 and end == 0):
                if isinstance(start, int):
                    add_ranges = [[start, end]]
                else:
                    add_ranges = list(zip(start, end))

                if rm_range is not None:
                    for add_range in add_ranges:
                        context_used_pos.extend(list(range(add_range[0], add_range[1])))
                        ret_ops.append(['replace', rm_range, add_range])
                    rm_range = None
                else:
                    for add_range in add_ranges:
                        if super_mode in ['before', 'both']:
                            ret_ops.append(['before', [pointer, pointer], add_range])
                        if super_mode in ['after', 'both']:
                            if pointer >= 1:
                                ret_ops.append(['after', [pointer - 1, pointer - 1], add_range])
        elif diff_op == '=':
            if rm_range is not None:
                ret_ops.append(['remove', rm_range, []])
            start, end = sub_finder(current_seq, diff_content, current_used_pos
                                    )
            current_used_pos.extend(list(range(start, end)))
            rm_range = None
            pointer = end
    return ret_ops


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
