import os
import csv
import json
import yaml
import numpy as np
import scipy.stats
from tabulate import tabulate
from collections import OrderedDict

from sven.constant import CWES_DICT, VAL_SCENARIOS

class SecEvalRun:
    TOP_K = [1, 5, 10]
    def __init__(self, eval_dir, eval_type, vul_types, split):
        self.results = OrderedDict()
        for cwe in vul_types:
            with open(os.path.join(eval_dir, cwe, 'result.jsonl')) as f:
                lines = f.readlines()
            for line in lines:
                j = json.loads(line)
                scenario = (cwe, j['scenario'])
                if eval_type in ('trained', 'trained_subset'):
                    if split == 'val' and scenario not in VAL_SCENARIOS:
                        continue
                    elif split == 'test' and scenario in VAL_SCENARIOS:
                        continue
                if scenario not in self.results:
                    self.results[scenario] = OrderedDict()
                self.results[scenario][j['control']] = j

                scores_path = os.path.join(eval_dir, cwe, j['scenario'], j['control']+'_scores.json')
                if os.path.exists(scores_path):
                    with open(scores_path) as f:
                        scores_j = json.load(f)
                    sorted_scores_j = list(sorted(scores_j.items(), reverse=True, key=lambda i:i[1]))
                    sorted_progs = list([i[0] for i in sorted_scores_j])
                    codeql_path = os.path.join(eval_dir, cwe, j['scenario'], j['control']+'_codeql.csv')
                    with open(codeql_path) as f:
                        reader = csv.reader(f)
                        vuls = set()
                        for row in reader:
                            vuls.add(row[4].replace('/', ''))
                    gens = set(scores_j.keys())
                    secs = gens - vuls
                    for k in self.TOP_K:
                        num_sec = len(secs & set(sorted_progs[:k]))
                        num_gen = min(k, len(gens))
                        j[f'sec_rate_{k}'] = num_sec / num_gen * 100

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h

class SecEval:
    KEYS = ['sec_rate', 'sec', 'total', 'dup', 'non_parsed']

    def __init__(self, eval_dir, eval_type, vul_type, split):
        if vul_type is not None:
            vul_types = [vul_type]
        else:
            vul_types = CWES_DICT[eval_type]

        self.runs = []
        if os.path.exists(os.path.join(eval_dir, eval_type)):
            self.runs.append(SecEvalRun(os.path.join(eval_dir, eval_type), eval_type, vul_types, split))
        else:
            for seed in os.listdir(eval_dir):
                eval_dir_seed = os.path.join(eval_dir, seed, eval_type)
                if os.path.isdir(eval_dir_seed):
                    self.runs.append(SecEvalRun(eval_dir_seed, eval_type, vul_types, split))

        detail_results = OrderedDict()
        overall_results = OrderedDict()
        for run in self.runs:
            overall_result = OrderedDict()
            for scenario in run.results:
                if scenario not in detail_results:
                    detail_results[scenario] = OrderedDict()
                for control in run.results[scenario]:
                    if control not in detail_results[scenario]:
                        detail_results[scenario][control] = OrderedDict()
                    if control not in overall_result:
                        overall_result[control] = OrderedDict()
                    for key in self.KEYS:
                        if key not in run.results[scenario][control] and key != 'sec_rate':
                            continue
                        if key not in detail_results[scenario][control]:
                            detail_results[scenario][control][key] = list()
                        if key not in overall_result[control]:
                            overall_result[control][key] = list()
                        
                        if key == 'sec_rate':
                            if run.results[scenario][control]['total'] != 0:
                                value = run.results[scenario][control]['sec'] / run.results[scenario][control]['total'] * 100
                                detail_results[scenario][control][key].append(value)
                                overall_result[control][key].append(value)
                        else:
                            detail_results[scenario][control][key].append(run.results[scenario][control][key])
                            overall_result[control][key].append(run.results[scenario][control][key])
            for control in overall_result:
                if control not in overall_results:
                    overall_results[control] = OrderedDict()
                for key in self.KEYS:
                    if key not in overall_result[control]:
                        continue
                    if key not in overall_results[control]:
                        overall_results[control][key] = list()
                    overall_results[control][key].append(np.mean(overall_result[control][key]))

        self.detail_results = detail_results
        self.overall_results = overall_results

    def format_stats_to_row(self, key, stats):
        mean, ci_low, ci_high = stats
        if key.startswith('sec_rate'):
            s_mean = '{:.1f},'.format(mean).ljust(7)
            s_ci_low = '{:.1f},'.format(mean - ci_low).ljust(7)
            s_ci_high = '{:.1f}'.format(mean + ci_high).rjust(6)
            return '{} {} {}'.format(s_mean, s_ci_low, s_ci_high)
        else:
            return '{:.1f}'.format(mean)

    def get_stats(self, values):
        mean = np.mean(values)
        ci_low, ci_high = confidence_interval(values)
        ci_low = ci_low if ci_low > 0 else 0.0
        ci_low = mean - ci_low
        ci_high = ci_high if ci_high < 100 else 100.0
        ci_high = ci_high - mean
        return mean, ci_low, ci_high

    def pretty_print(self):
        table = []
        for scenario in self.detail_results:
            for control in self.detail_results[scenario]:
                row = [scenario[0], scenario[1], control]
                for key in self.detail_results[scenario][control]:
                    values = self.detail_results[scenario][control][key]
                    stats = self.get_stats(values)
                    row.append(self.format_stats_to_row(key, stats))
                table.append(row)

        for control in self.overall_results:
            row = ['overall', 'overall', control]
            for key in self.overall_results[control]:
                values = self.overall_results[control][key]
                stats = self.get_stats(values)
                row.append(self.format_stats_to_row(key, stats))
            table.append(row)

        headers = ['cwe', 'scenario', 'control']
        for key in self.overall_results[list(self.overall_results.keys())[0]]:
            if key.startswith('sec_rate'):
                header = key + ': {}   {} {}'.format('mean,', 'ci_low,', 'ci_high')
            else:
                header = key + ': mean'
            headers.append(header)

        print(tabulate(table, headers=headers, stralign='right', tablefmt='orgtbl'))

    def get_overall_sec_rate(self):
        res = OrderedDict()
        for control in self.overall_results:
            res[control] = OrderedDict()
            for key in self.KEYS:
                if not key.startswith('sec_rate'): continue
                if key not in self.overall_results[control]: continue
                values = self.overall_results[control][key]
                stats = self.get_stats(values)
                res[control][key] = stats
        return res

def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

class FuncEval:
    K = [1, 5, 10, 25, 50, 100]

    def __init__(self, eval_dir):
        self.pass_k = [[] for _ in range(len(self.K))]
        for fname in os.listdir(eval_dir):
            if not fname.endswith('.results.yaml'): continue
            with open(os.path.join(eval_dir, fname)) as f:
                res_data = yaml.load(f, Loader=yaml.CLoader)
            n, c = 0, 0
            for r in res_data['results']:
                n += 1
                if r['status'] == 'OK':
                    c += 1
            for i, k in enumerate(self.K):
                self.pass_k[i].append(pass_at_k(n, c, k))
        for i, k in enumerate(self.K):
            self.pass_k[i] = np.mean(self.pass_k[i])*100

    def pretty_print(self):
        header, row = [], []
        for i, k in enumerate(self.K):
            header.append(f'pass@{k}')
            row.append('{:.1f}'.format(self.pass_k[i]))
        print(tabulate([row], headers=header, stralign='right', tablefmt='orgtbl'))

    def get_pass_k(self):
        res = OrderedDict()
        for i, k in enumerate(self.K):
            res[f'pass@{k}'] = self.pass_k[i]
        return res