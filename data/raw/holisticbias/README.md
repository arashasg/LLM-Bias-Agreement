---
configs:
  - config_name: noun_phrases
    data_files:
    - split: test
      path: nouns.csv
  - config_name: sentences
    data_files:
    - split: test
      path: sentences.csv
license: cc-by-sa-4.0
language: en
---

# Usage

When downloading, specify which files you want to download and set the split to `train` (required by `datasets`).

```python
from datasets import load_dataset

nouns = load_dataset("fairnlp/holistic-bias", data_files=["nouns.csv"], split="train")
sentences = load_dataset("fairnlp/holistic-bias", data_files=["sentences.csv"], split="train")
```

# Dataset Card for Holistic Bias

This dataset contains the source data of the Holistic Bias dataset as described [by Smith et. al. (2022)](https://arxiv.org/abs/2205.09209). The dataset contains noun phrases and sentences used to measure the likelihood bias of various models. The original dataset is released on [GitHub](https://github.com/facebookresearch/ResponsibleNLP/tree/main/holistic_bias).

Disclaimer: this re-release of the dataset is not associated with the original authors. The dataset is released under the [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

## Dataset Details

The data is generated from the [official generation script](https://github.com/facebookresearch/ResponsibleNLP/blob/main/holistic_bias/generate_sentences.py). The data is the v1.0 data from the original paper.

For details on the methodology, please refer to the original paper. This dataset is contributed to Hugging
Face as part of the [FairNLP `fairscore` library](https://github.com/FairNLP/fairscore/).

### Dataset Sources

- **Paper:** https://arxiv.org/pdf/2205.09209.pdf

**BibTeX:**

```bibtex
@inproceedings{smith2022m,
  title={“I’m sorry to hear that”: Finding New Biases in Language Models with a Holistic Descriptor Dataset},
  author={Smith, Eric Michael and Hall, Melissa and Kambadur, Melanie and Presani, Eleonora and Williams, Adina},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages={9180--9211},
  year={2022}
}
```