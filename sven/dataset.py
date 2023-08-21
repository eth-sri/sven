import os
import abc
import json
import torch
import random
from torch.utils.data import Dataset

from sven.constant import BINARY_LABELS, SEC_LABEL, VUL_LABEL, PROMPTS, CWES_TRAINED, CWES_TRAINED_SUBSET
from sven.utils import get_indent

class DatasetBase(Dataset):
    def __init__(self, args, tokenizer, mode):
        self.args = args
        self.tokenizer = tokenizer
        self.dataset = list()
        if self.args.vul_type is not None:
            vul_types = [self.args.vul_type]
        else:
            if 'incoder' in self.args.pretrain_dir:
                vul_types = CWES_TRAINED_SUBSET
            else:
                vul_types = CWES_TRAINED
        for i, vul_type in enumerate(vul_types):
            with open(os.path.join(args.data_dir, mode, f'{vul_type}.jsonl')) as f:
                lines = f.readlines()
            for line in lines:
                diff_j = json.loads(line)
                if diff_j['file_name'].endswith('.py'):
                    lang = 'py'
                else:
                    lang = 'c'
                labels = [SEC_LABEL, VUL_LABEL]
                srcs = [diff_j['func_src_after'], diff_j['func_src_before']]
                if self.args.diff_level == 'prog':
                    diffs = [None, None]
                elif self.args.diff_level == 'line':
                    diffs = [diff_j['line_changes']['added'], diff_j['line_changes']['deleted']]
                elif self.args.diff_level == 'char':
                    diffs = [diff_j['char_changes']['added'], diff_j['char_changes']['deleted']]
                elif self.args.diff_level == 'mix':
                    diffs = [diff_j['char_changes']['added'], diff_j['line_changes']['deleted']]
                else:
                    raise NotImplementedError()
                for label, src, changes in zip(labels, srcs, diffs):
                    self.add_data(label, src, changes, i, lang)

    @abc.abstractclassmethod
    def add_data(self, label, src, changes, vul_id):
        raise NotImplementedError()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return tuple(torch.tensor(t) for t in self.dataset[item])

class PrefixDataset(DatasetBase):
    def __init__(self, args, tokenizer, mode):
        super().__init__(args, tokenizer, mode)

    def add_data(self, label, src, changes, vul_id, lang):
        control_id = BINARY_LABELS.index(label)    
        data = self.get_tensor(src, vul_id, control_id, changes)
        if data is not None:
            self.dataset.append(data)

    def get_tensor(self, src, vul_id, control_id, changes):
        be = self.tokenizer.encode_plus(src)
        tokens = be.data['input_ids']
        if len(tokens) > self.args.max_num_tokens: return None

        min_changed_tokens = (2 if self.args.vul_type in ('cwe-invalid', 'cwe-valid') else 1)
        if changes is None:
            weights = [1] * len(tokens)
        else:
            weights = [0] * len(tokens)
            for change in changes:
                char_start = change['char_start']
                char_start_idx = be.char_to_token(char_start)
                char_end = change['char_end']
                char_end_idx = be.char_to_token(char_end-1)
                for char_idx in range(char_start_idx, char_end_idx+1):
                    weights[char_idx] = 1
            if sum(weights) < min_changed_tokens: return None
            if len(tokens) - sum(weights) < min_changed_tokens: return None

        return tokens, weights, control_id, vul_id

class TextPromptDataset(DatasetBase):
    def __init__(self, args, tokenizer, mode):
        super().__init__(args, tokenizer, mode)

    def add_data(self, label, src, changes, vul_id, lang):
        control_id = BINARY_LABELS.index(label)    
        if lang == 'py':
            control = get_indent(src) + '# ' + PROMPTS[control_id]
        else:
            control = get_indent(src) + '// ' + PROMPTS[control_id]
        src = control + src
        data = self.get_tensor(src, control, changes)
        if data is not None:
            self.dataset.append(data)

    def get_tensor(self, src, control, changes):
        be = self.tokenizer.encode_plus(src)
        tokens = be.data['input_ids']

        if changes is None:
            labels = tokens[:]
        else:
            labels = [-100] * len(tokens)
            label_set = False
            for change in changes:
                char_start = change['char_start'] + len(control)
                char_start_idx = be.char_to_token(char_start)
                char_end = change['char_end'] + len(control)
                char_end_idx = be.char_to_token(char_end-1)
                for i in range(char_start_idx, char_end_idx+1):
                    labels[i] = tokens[i]
                    label_set = True
            if not label_set: return None

        if len(tokens) > self.args.max_num_tokens: return None
        return tokens, labels