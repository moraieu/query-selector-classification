from argparse import Namespace

import numpy as np
from torch.utils.data import Dataset
import torch
from code.load_data import load_cases, get_dataset_params, get_training_cases
from query_selector.model import Transformer

additional_fields = 28 + 1 + 15


def build_model(target_chars, maxlen, heads, dim_val, dim_attn, attn_type):
    return Transformer(dim_val=dim_val, dim_attn=dim_attn, input_size=len(target_chars) + additional_fields,
                       dec_seq_len=maxlen, out_seq_len=len(target_chars), output_len=1,
                       n_heads=heads, n_encoder_layers=2, n_decoder_layers=1,
                       enc_attn_type=attn_type, dec_attn_type="full", dropout=0.4)
    return get_model(args)


def encode_helpdesk(line, datetimes, maxlen, target_chars, target_char_indices):
    input_tensor = torch.zeros([1, maxlen, len(target_chars) + additional_fields], device="cuda")
    leftpad = maxlen - len(line)
    for i, c in enumerate(line):
        # action embedding
        input_tensor[0, leftpad + i, target_char_indices[c]] = 1
        # weekday embedding
        input_tensor[0, leftpad + i, len(target_chars) + (datetimes[i].weekday() % 3)] = 1
        # year embedding
        input_tensor[0, leftpad + i, len(target_chars) + 3 + datetimes[i].year - 2010] = 1
        # month embedding
        input_tensor[0, leftpad + i, len(target_chars) + 8 + datetimes[i].month - 1] = 1
        # hour embedding
        input_tensor[0, leftpad + i, len(target_chars) + 20 + (datetimes[i].hour // 3)] = 1
        # call len embedding
        input_tensor[0, leftpad + i, len(target_chars) + 28 + len(line)] = 1
        # input_tensor[0, leftpad + i, -2] = 0 if i == 0 else (datetimes[i] - datetimes[i-1]).total_seconds() / (7*3600*24)
        input_tensor[0, leftpad + i, -1] = 0 if i == 0 or (datetimes[i].date != datetimes[i - 1].date) else 1
        # print( datetimes[i], datetimes[i].weekday())
    return input_tensor


def encode_bpi(line, datetimes, maxlen, target_chars, target_char_indices):
    input_tensor = torch.zeros([1, maxlen, len(target_chars) + additional_fields], device="cuda")
    leftpad = maxlen - len(line)
    for i, c in enumerate(line):
        # action embedding
        input_tensor[0, leftpad + i, target_char_indices[c]] = 1
        # weekday embedding
        input_tensor[0, leftpad + i, len(target_chars) + (datetimes[i].weekday() % 3)] = 1
        # year embedding
        input_tensor[0, leftpad + i, len(target_chars) + 3 + datetimes[i].year - 2010] = 1
        # month embedding
        input_tensor[0, leftpad + i, len(target_chars) + 8 + datetimes[i].month - 1] = 1
        # hour embedding
        input_tensor[0, leftpad + i, len(target_chars) + 20 + (datetimes[i].hour // 3)] = 1
        # call len embedding
        # input_tensor[0, leftpad + i, len(target_chars) + 28 + len(line)] = 1
        input_tensor[0, leftpad + i, -3] = 0 if i == 0 else (datetimes[i] - datetimes[0]).total_seconds() / (
                7 * 3600 * 24)
        input_tensor[0, leftpad + i, -2] = 0 if i == 0 else (datetimes[i] - datetimes[i - 1]).total_seconds() / (
                    7 * 3600 * 24)
        input_tensor[0, leftpad + i, -1] = 0 if i == 0 or (datetimes[i].date != datetimes[i - 1].date) else 1
        # print( datetimes[i], datetimes[i].weekday())

    # print(input_tensor)
    return input_tensor


class BpiDataset(Dataset):
    def __init__(self, path):
        cases = load_cases(path)
        elems_per_fold, self.maxlen, chars, self.target_chars, char_indices, self.target_char_indices, target_indices_char = \
            get_dataset_params(cases)

        cases = get_training_cases(path)

        print(len(self.target_chars))

        for case in cases:
            case.line = case.line + '!'

        self.train_cases = []

        for case in cases:
            line = case.line

            for i in range(1, len(line)):
                self.train_cases.append([case.trim(i), case.extract_event(i)])

    def __getitem__(self, index):
        out_tensor = torch.zeros([1, len(self.target_chars), 1], device="cuda")
        out_tensor[0, self.target_char_indices[self.train_cases[index][1].event_char], 0] = 1
        input_tensor = encode_bpi(self.train_cases[index][0].line, self.train_cases[index][0].datetimes, self.maxlen,
                                  self.target_chars, self.target_char_indices)
        return input_tensor.squeeze(0), out_tensor.squeeze(0)

    def __len__(self):
        return len(self.train_cases)


class HelpdeskDataset(BpiDataset):
    def __getitem__(self, index):
        out_tensor = torch.zeros([1, len(self.target_chars), 1], device="cuda")
        out_tensor[0, self.target_char_indices[self.train_cases[index][1].event_char], 0] = 1
        input_tensor = encode_helpdesk(self.train_cases[index][0].line, self.train_cases[index][0].datetimes,
                                       self.maxlen, self.target_chars, self.target_char_indices)
        return input_tensor.squeeze(0), out_tensor.squeeze(0)
