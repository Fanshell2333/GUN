#!/user/bin/env python3
# -*- coding: utf-8 -*-
import re
import numpy as np
from typing import List
from util.data_util import SpecialSymbol, export_word_edit_matrix, get_class_mapping
from transformers import BertTokenizerFast


class Processor:
    def __init__(self, train_path: str = None,
                 dev_path: str = None,
                 test_path: str = None,
                 tokenizer: str = 'bert-base-chinese',
                 super_mode: str = 'before',
                 enable_unparse: bool = True,
                 extra_stop_words: List = None,
                 joint_encoding: bool = True):
        if not train_path and dev_path and test_path:
            raise Warning("Because of there are no data path, we can only use build label for one case")

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer)
        self._super_mode = super_mode
        self._enable_unparse = enable_unparse
        self._extra_stop_words = extra_stop_words
        self._joint_encoding = joint_encoding

    def build_data(self, context_utt: List[str],
                   cur_utt: str,
                   restate_utt: str,
                   training: bool = True):
        if self._extra_stop_words is not None and len(self._extra_stop_words) > 0:
            context_utt = [' '.join(self._extra_stop_words)] + context_utt

        context_utt = re.sub('\\s+', ' ', ' '.join([sen.lower() + ' ' + SpecialSymbol.context_internal
                                                    for sen in context_utt]))
        cur_utt = re.sub('\\s+', ' ', cur_utt.lower())
        restate_utt = re.sub('\\s+', ' ', restate_utt.lower())

        tokenized_context = self.tokenizer.tokenize(context_utt)
        tokenized_cur = self.tokenizer.tokenize(cur_utt.lower())
        tokenized_cur.append(SpecialSymbol.end_placeholder)
        tokenized_restate = self.tokenizer.tokenize(restate_utt)

        if self._joint_encoding:
            tokenized_joint = tokenized_context + tokenized_cur

        if self._extra_stop_words:
            # maybe not reasonable
            attn_operations = export_word_edit_matrix(tokenized_context,
                                                      tokenized_cur[:-1],
                                                      tokenized_restate,
                                                      self._super_mode,
                                                      only_one_insert=False)
        else:
            attn_operations = export_word_edit_matrix(tokenized_context,
                                                      tokenized_cur[:-1],
                                                      tokenized_restate,
                                                      self._super_mode,
                                                      only_one_insert=True)

        matrix_map = np.zeros((len(tokenized_context), len(tokenized_cur)),
                              dtype=np.long)

        class_mapping = get_class_mapping(super_mode=self._super_mode)

        # build distant supervision
        if training:
            keys = [op_tuple[0] for op_tuple in attn_operations]
            if not self._enable_unparse and 'remove' in keys:
                # the training supervision may not be accurate
                print("Invalid Case")
            else:
                for op_tuple in attn_operations:
                    op_name = op_tuple[0]

                    if op_name == 'remove':
                        continue

                    assert op_name in class_mapping.keys()
                    label_value = class_mapping[op_name]

                    cur_start, cur_end = op_tuple[1]
                    con_start, con_end = op_tuple[2]
                    if op_name == 'replace':
                        matrix_map[con_start:con_end, cur_start:cur_end] = label_value
                    else:
                        assert cur_start == cur_end
                        matrix_map[con_start:con_end, cur_start] = label_value

        return tokenized_context, matrix_map


def main():
    processor = Processor()
    context_utt = ['#', '打 开 车 窗 关 闭 空 调']
    cur_utt = '打 开 车 窗 关 闭 空 调'
    restate_utt = '打 开 车 窗 # 关 闭 空 调'

    context_utt = ['#', '打 开 窗 关 闭 空 调']
    cur_utt = '打 开 窗 关 闭 空 调'
    restate_utt = '打 开 窗 # 关 闭 空 调'

    uttr, attn_matrix = processor.build_data(context_utt, cur_utt, restate_utt)
    print(uttr)
    print(attn_matrix)


if __name__ == '__main__':
    main()
