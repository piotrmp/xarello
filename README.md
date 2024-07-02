#🍷 XARELLO: eXploring Adversarial examples using REinforcement Learning Optimisation

This repository contains the code of XARELLO: a solution for finding adversarial examples and thus testing the robustness of text classifiers,
especially in the tasks of misinformation detection. Unlike other solutions, XARELLO is *adaptive*, i.e. it observes the responses from the
victim classifier and learns, which attacks are succesfull in changing its decisions. This adaptation process is performed using the
reinforcement learning, implemented as Q-learning with Transformer-based Q estimator.

The full description of the solution, including the evaluation results, is available in the article [Verifying the Robustness of Automatic Credibility Assessment](TODO),
presented at the [WASSA](https://workshop-wassa.github.io) workshop at ACL 2024. 

The research was done within the [ERINIA](https://www.upf.edu/web/erinia) project realised at the
[TALN lab](https://www.upf.edu/web/taln/) of [Universitat Pompeu Fabra](https://www.upf.edu).

## Installation

## Usage

## Licence

XARELLO code is released under the [GNU GPL 3.0](https://www.gnu.org/licenses/gpl-3.0.html) licence.

## Funding

The [ERINIA](https://www.upf.edu/web/erinia) project has received funding from the European Union’s Horizon Europe
research and innovation programme under grant agreement No 101060930.

Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the
European Union. Neither the European Union nor the granting authority can be held responsible for them.
