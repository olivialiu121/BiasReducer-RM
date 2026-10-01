# BiasReducer: Adaptive Bias Mitigation for Reward Models

Code for [BiasReducer: Adaptive Bias Mitigation for Reward Models](https://arxiv.org/abs/2609.32720).

BiasReducer edits only the linear head of a trained reward model to reduce its
dependence on superficial response attributes (length, verbosity, formatting,
emoji/exclamation use, confidence, hedging, sycophancy), and selects which
attributes to correct separately for every new dataset, using the dataset's
responses and original rewards but no preference labels.

1. **Learn** (`biasreducer/representation/`): a sparse encoder-decoder over source
   preference contrasts with one designated coordinate per attribute, trained with
   behavioral and intervention supervision.
2. **Calibrate** (`biasreducer/edit/`): for each attribute, an edit direction and
   an edit amount chosen under a source-validation accuracy budget; together they
   form the edit bank.
3. **Select and edit** (`biasreducer/select/`): on a new dataset, rank attributes
   by reward dependence and edit impact; **BiasReducer-S** applies the top-ranked
   edit, **BiasReducer-M** composes several.

## Setup

```bash
pip install -e .
```

A GPU is needed for the forward passes through the reward model; everything
else runs on CPU. Hyperparameters are in `configs/biasreducer.yaml`, the
attributes in `configs/attributes.yaml`, the benchmarks in
`configs/benchmarks.yaml` and the reward models in `configs/models/`.

## Pipeline

```bash
M=skywork_qwen3_1p7b          # any configs/models/*.yaml stem

# benchmarks (JudgeBiasBench: set benchmarks.judgebiasbench.source_dir first)
python -m scripts.prepare_benchmark --benchmark rmbench_hard        --models $M
python -m scripts.prepare_benchmark --benchmark arena_styleconflict --models $M
python -m scripts.prepare_benchmark --benchmark judgebiasbench      --models $M

python -m scripts.extract_source       --model $M   # source pairs -> hidden states
python -m scripts.build_supervision    --model $M   # rewrites, measurements, artifact direction
python -m scripts.learn_representation --model $M   # stage 1
python -m scripts.build_edit_bank      --model $M   # stage 2
python -m scripts.select_and_edit      --model $M --benchmark rmbench_hard   # stage 3
```

Results are written to `artifacts/results/<model>/<benchmark>.json`: the
selection signals, the selected edits and the pairwise preference accuracy of
the original, BiasReducer-S and BiasReducer-M heads.

## Citation

```bibtex
@misc{liu2026biasreducer,
  title={BiasReducer: Adaptive Bias Mitigation for Reward Models},
  author={Shuang Liu and Yongliang Miao and Yanguang Liu and Haoyi Xiong and Mengnan Du},
  year={2026},
  eprint={2609.32720},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2609.32720},
}
```
