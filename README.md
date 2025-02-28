# Benchmarking with a Language Model Initial Selection for Text Classification Tasks
This study explores the effectiveness of language model initial selection for text classification tasks. The paper presents a benchmarking framework and evaluates the performance of various language models in different classification scenarios.

## Overview

Current benchmarking methods often rely on inefficient "brute force" testing of all candidate models, which is environmentally harmful. To address this, the study introduces Language Model-Dataset Fit (LMDFit), an innovative benchmarking approach that incorporates an initial model selection process. LMDFit eliminates underperforming models early by evaluating them on a proxy task (semantic similarity assessment) using a small dataset, significantly reducing computational costs and emissions. Inspired by organizational hiring practices, LMDFit ensures only relevant models undergo extensive testing. Experiments across eight text classification tasks and seven pre-trained language models demonstrate LMDFit's efficiency, reducing benchmarking time and emissions by 37% on average compared to conventional methods. This approach promotes sustainable and environmentally friendly AI development.

## Citation
If you use **LMDFit** in your research or work, please cite the following paper:

**APA Style**

Riyadi, A., Kovacs, M., Serdült, U., & Kryssanov, V. (2025). Benchmarking with a Language Model Initial Selection for Text Classification Tasks. Machine Learning and Knowledge Extraction, 7(1), 3. https://doi.org/10.3390/make7010003

**Bibtex**
```bibtex
@article{riyadi2025benchmarking,
  title={Benchmarking with a Language Model Initial Selection for Text Classification Tasks},
  author={Riyadi, Agus and Kovacs, Marton and Serdült, Uwe and Kryssanov, Victor},
  journal={Machine Learning and Knowledge Extraction},
  volume={7},
  number={1},
  pages={3},
  year={2025},
  doi={10.3390/make7010003}
}
