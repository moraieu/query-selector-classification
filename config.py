import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', type=str, default="data/helpdesk.csv")
    parser.add_argument('--cls_iterations', type=int, default=6)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=144)
    parser.add_argument('--dim_val', type=int, default=96)
    parser.add_argument('--dim_attn', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--attn_type', type=str, default="query_selector_0.9")

    return parser