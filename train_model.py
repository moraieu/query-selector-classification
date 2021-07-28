from argparse import Namespace

from code.load_data import load_cases, get_dataset_params, get_training_cases
from query_selector.model import Transformer
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from config import get_arg_parser
from model_and_datasets import BpiDataset, HelpdeskDataset, build_model

args = get_arg_parser().parse_args()

cases = load_cases(args.data)

elems_per_fold, maxlen, chars, target_chars, char_indices, target_char_indices, target_indices_char = \
    get_dataset_params(cases)

cases = get_training_cases(args.data)

print(maxlen, len(target_chars))

for case in cases:
    case.line = case.line +'!'

train_cases = []

for case in cases:
    line = case.line
    line_t = case.times
    line_t2 = case.times2
    line_t3 = case.times3
    line_t4 = case.times4

    for i in range(1, len(line)):
        train_cases.append([ case.trim(i), case.extract_event(i)])


def get_params(mdl):
    return mdl.parameters()


m = build_model(target_chars, maxlen, heads=args.heads, dim_val=args.dim_val, dim_attn=args.dim_attn,
                attn_type=args.attn_type)
m.to('cuda')
m.optim = Adam(get_params(m), lr=args.lr)

hits = 0
tests = 0


if args.data.endswith("helpdesk.csv"):
    dataset = HelpdeskDataset(args.data)
else:
    dataset = BpiDataset(args.data)

train_data_cls_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


print("Training classifier...")
for it in tqdm(range(args.cls_iterations)):
    hits = 0
    tests = 0
    dst = tqdm(train_data_cls_loader)
    for input_tensor, out_tensor in dst:
        m.optim.zero_grad()
        result = m(input_tensor)
        result_indices = torch.max(result, dim=1).indices
        target_indices = torch.max(out_tensor, dim=1).indices
        hits += torch.sum(target_indices == result_indices).item()
        tests += input_tensor.shape[0]

        loss = torch.nn.CrossEntropyLoss()(result, target_indices)

        loss.backward()
        m.optim.step()
        dst.set_description("IT: {}, CrossEntropy: {}, accuracy: {}, cases: {}".format(it + 1, loss.item(), hits / tests, tests))

torch.save(m.state_dict(), "models/train_checkpoint.pt")