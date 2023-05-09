# SVEN: Security Hardening and Adversarial Testing for Code LLMs
SVEN enables controlling LLMs to generate secure (for security hardening) or unsafe code (for adversarial testing), while maintaining functional correctness. It achieves this by learning continuous prompts (or prefixes) with specialized loss terms on our curated dataset. SVEN currently supports [CodeGen LLMs](https://github.com/salesforce/CodeGen) but can be applicable to other LLMs. This repository contains SVEN's source code and trained prefixes, as well as training and evaluation data. For more technical details, check our [paper](https://arxiv.org/abs/2302.05319).

## Directory Structure
The directory structure of this repository is shown as below:
```
.
|-- data_train_val     # our curated dataset for training and validation (Table 1)
|-- data_eval          # datasets used for evaluation
    |-- dow            # evaluation scenarios for our main CWEs (Table 2)
    |-- dop            # evaluation scenarios for prompt perturbations (Figure 12)
    |-- not_trained    # evaluation scenarios for CWEs unseen during training (Table 3)
    |-- human_eval     # HumanEval benchmark from the MultiPL-E framework (Table 4)
|-- sven               # SVEN's source code
|-- scripts            # scripts for training and evaluation
|-- trained            # trained prefixes for 350M, 2.7B, and 6.1B codegen-multi models
```

## Setup
Clone this repository and set up Python dependencies (a virtual environment is recommended):
```console
$ pip install -r requirements.txt
$ pip install -e .
```

Set up [GitHub CodeQL](https://github.com/github/codeql):
```console
$ wget https://github.com/github/codeql-cli-binaries/releases/download/v2.11.1/codeql-linux64.zip
$ unzip codeql-linux64.zip
$ git clone --depth=1 --branch codeql-cli-2.11.1 https://github.com/github/codeql.git codeql/codeql-repo
$ codeql/codeql pack download codeql-cpp codeql-python@0.6.2 codeql/ssa codeql/tutorial codeql/regex codeql/util
$ cp data_eval/dow/cwe-190/1-c/ArithmeticTainted.ql codeql/codeql-repo/cpp/ql/src/Security/CWE/CWE-190/ArithmeticTainted.ql
```

## Evaluation
The evaluation consists of two parts: security and functional correctness. You should run the evaluation scripts under the `./scripts` directory. Make sure to use `CUDA_VISIBLE_DEVICES` to select the correct GPUs.

### Evaluation on Security
To evaluate the security of the original LLM, run the command below. The model size `350m` can be replaced with `2b` or `6b`. See `sec_eval.py` for other options, such as using `--temp` to adjust temperature and using `--eval_type` to select the evaluation scenarios.
```console
$ python sec_eval.py --model_type lm --model_dir 350m --output_name sec-eval-350m-lm
```

To evaluate the security of SVEN using the trained models provided by us, run:
```console
$ python sec_eval.py --model_type prefix --model_dir ../trained/350m-prefix/checkpoint-last --output_name sec-eval-350m-prefix
```

Use `print_results.py` to obtain the evaluation results. An example command for the original LLM is:
```console
$ python print_results.py --eval_dir ../experiments/sec_eval/sec-eval-350m-lm
```

### Evaluation on Functional Correctness
We use [the HumanEval benchmark](https://github.com/openai/human-eval) from [the MultiPL-E framework](https://github.com/nuprl/MultiPL-E/tree/dbcfa139a66cf5e46de798fa5e0854a7f417a046) to evaluate functional correctness. To evaluate the original LLM, run the command below. Check `human_eval_gen.py` for other generation arguments.
```console
$ python human_eval_gen.py --model_type lm --model_dir 350m --output_name human-eval-350m-lm
$ python human_eval_exec.py --output_name human-eval-350m-lm
```

For SVEN, we need to run the two branches `sec` and `vul` separately via the `--control` argument. The command below is for the `sec` branch:
```console
$ python human_eval_gen.py --model_type prefix --model_dir ../trained/350m-prefix/checkpoint-last --control sec --output_name human-eval-350m-prefix-sec
$ python human_eval_exec.py --output_name human-eval-350m-prefix-sec
```

To view the results (for the original LLM for example), run:
```console
$ python print_results.py --eval_type human_eval --eval_dir ../experiments/human_eval/human-eval-350m-lm
```

## Training
We have provided our trained models in `./trained`. To train SVEN yourself, run:
```console
$ python train.py --output_name 350m-prefix-new --pretrain_dir 350m
```

## Citation
```
@article{sven-llm,
  author    = {Jingxuan He and Martin Vechev},
  title     = {Large Language Models for Code: Security Hardening and Adversarial Testing},
  journal   = {CoRR},
  volume    = {abs/2302.05319},
  year      = {2023},
  url       = {https://arxiv.org/abs/2302.05319},
}
```