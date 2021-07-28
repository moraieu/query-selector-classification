from argparse import Namespace

import numpy as np
import torch
from query_selector.model import Transformer
from code.load_data import load_cases, valid_cases_generator, get_dataset_params
from config import get_arg_parser
from model_and_datasets import encode_helpdesk, encode_bpi, build_model
args = get_arg_parser().parse_args()
cases = load_cases(args.data)

elems_per_fold, maxlen, chars, target_chars, char_indices, target_char_indices, target_indices_char = \
    get_dataset_params(cases)


if args.data.endswith("helpdesk.csv"):
    encoder = encode_helpdesk
else:
    encoder = encode_bpi


def get_params(mdl):
    return mdl.parameters()


m = build_model(target_chars, maxlen, heads=args.heads, dim_val=args.dim_val, dim_attn=args.dim_attn,
                attn_type=args.attn_type)

m.load_state_dict(torch.load("models/train_checkpoint.pt"))
m.to('cuda')
m.eval()


hits = 0
cases = 0

for _, prefix_size, cropped_line, cropped_times, cropped_times3, ground_truth, _ \
        in valid_cases_generator(args.data, maxlen):
    print(cropped_line, "->", ground_truth)
    input_tensor = encoder(cropped_line, cropped_times3, maxlen, target_chars, target_char_indices)
    result = m(input_tensor)
    prediction = target_chars[torch.argmax(result)]
    cases += 1
    if prediction == '!' or len(ground_truth) == 0:
        if prediction == '!' and len(ground_truth) == 0:
            hits += 1
    else:
        if prediction == ground_truth[0]:
            hits += 1
    print("Accuracy ", hits / cases)