# SVEN: Security Hardening and Adversarial Testing for Code LLMs
SVEN enables controlling LLMs to generate secure (for security hardening) or unsafe code (for adversarial testing), while maintaining functional correctness. It achieves this by learning continuous prompts (or prefixes) with specialized loss terms on our curated dataset. This repository contains SVEN's source code and trained prefixes, as well as training and evaluation data. For more technical details, check our [paper](https://arxiv.org/abs/2302.05319).

## Directory Structure
The directory structure of this repository is shown as below:
```
.
|-- data_train_val     # our curated dataset for training and validation
|-- data_eval          # datasets used for evaluation
|-- sven               # SVEN's source code
|-- scripts            # scripts for training and evaluation
|-- trained            # trained prefixes
```

SVEN currently supports [CodeGen](https://arxiv.org/abs/2203.13474), [InCoder](https://arxiv.org/abs/2204.05999), and [SantaCoder](https://arxiv.org/abs/2301.03988). It should be straightforward to add support for other LLMs (PR welcomed).

## Setup
Set up Python dependencies (a virtual environment is recommended) and [GitHub CodeQL](https://github.com/github/codeql):
```console
$ pip install -r requirements.txt
$ pip install -e .
$ ./setup_codeql.sh
```

## Evaluation
The evaluation consists of two parts: security and functional correctness. You should run the evaluation scripts under the `./scripts` directory. Make sure to use `CUDA_VISIBLE_DEVICES` to select the correct GPUs.

### Evaluation on Security
To evaluate the security of the original LLM, run the command below. The model `350m` can be replaced by {`2b`, `6b`, `incoder`, `santa`}. See `sec_eval.py` for other options, such as using `--temp` to adjust temperature and using `--eval_type` to select the evaluation scenarios.
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
We have provided our trained prefixes in `./trained`. To train SVEN yourself, run:
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