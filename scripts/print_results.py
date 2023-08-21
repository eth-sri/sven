import os
import json
import argparse

from sven.metric import SecEval, FuncEval

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', type=str, choices=['trained', 'trained_subset', 'prompts', 'gen_1', 'gen_2', 'human_eval'], default='trained')
    parser.add_argument('--eval_dir', type=str, required=True)
    parser.add_argument('--vul_type', type=str, default=None)
    parser.add_argument('--split', type=str, choices=['val', 'test', 'all'], default='test')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    if args.eval_type == 'human_eval':
        e = FuncEval(args.eval_dir)
    else:
        e = SecEval(args.eval_dir, args.eval_type, args.vul_type, args.split)
    e.pretty_print()

if __name__ == '__main__':
    main()