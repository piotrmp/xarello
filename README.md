# 🍷 XARELLO: eXploring Adversarial examples using REinforcement Learning Optimisation🍷

This repository contains the code of XARELLO: a solution for finding adversarial examples and thus testing the robustness of text classifiers,
especially in the tasks of misinformation detection. Unlike other solutions, XARELLO is *adaptive*, i.e. it observes the responses from the
victim classifier and learns, which attacks are succesfull in changing its decisions. This adaptation process is performed using
reinforcement learning, implemented as Q-learning with Transformer-based Q estimator.

The full description of the solution, including the evaluation results, is available in the article [Verifying the Robustness of Automatic Credibility Assessment](TODO),
presented at the [WASSA](https://workshop-wassa.github.io) workshop at ACL 2024. 

The research was done within the [ERINIA](https://www.upf.edu/web/erinia) project realised at the
[TALN lab](https://www.upf.edu/web/taln/) of [Universitat Pompeu Fabra](https://www.upf.edu).

## Installation

To use XARELLO, you will first need to install [BODEGA](https://github.com/piotrmp/BODEGA) -- a benchmark for robustness testing in credibility assessment.
Additionally, XARELLO requires [gymnasium](https://github.com/Farama-Foundation/Gymnasium) and [fasttext](https://pypi.org/project/fasttext/). Using CONDA, 
an environment could be prepared through the following:
```commandline
conda create xarello
conda activate xarello
conda install python=3.10
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -U "transformers==4.38.1"
pip install git+https://gitlab.clarin-pl.eu/syntactic-tools/lambo.git
pip install OpenAttack
pip install editdistance
pip install bert-score
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
pip install peft bitsandbytes accelerate
pip install gymnasium
pip install fasttext
```
Note that BODEGA source will have to be in your PYTHONPATH for XARELLO to run properly.

## Usage

## Licence

XARELLO code is released under the [GNU GPL 3.0](https://www.gnu.org/licenses/gpl-3.0.html) licence.

## Funding

The [ERINIA](https://www.upf.edu/web/erinia) project has received funding from the European Union’s Horizon Europe
research and innovation programme under grant agreement No 101060930.

Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the
European Union. Neither the European Union nor the granting authority can be held responsible for them.
