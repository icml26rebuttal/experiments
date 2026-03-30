# Code for "Ain't no frequency high (low) enough: adversarial robustness through the lens of structured transforms" -  ICML'26 submission
Experimental results for the rebuttal of the paper "Ain't no frequency high (low) enough: adversarial robustness through the lens of structured transforms". The 2 main scripts are: 

a) ``eval_transferability.py`` -> generates results for the Tables 1-4 of the manuscript, as well as the additional tables mentioned in the rebuttal, for the CIFAR10. It is called as ``python eval_transferability.py --case case --model-source class``, where case lies in {case1, case2, case3, case4}, and class lies in {pretrained, robustbench}. Case1 = proposed DGF-PGD, case2 = standard PGD, case3 = F-PGD, case4 = AutoPGD. 

b) ``eval_transferability_cifar100.py`` -> same as ``eval_transferability.py``, but for CIFAR100.

Auxiliary scripts called by ``eval_transferability.py`` and ``eval_transferability_cifar100.py``:

a) ``dgf_pgd.py`` -> the class of adversarial attacks of cases 1-3

b) ``transforms`` -> implements all transforms appearing in the paper, along with the projection operator and the bisection method

c) ``evaluations_metrics.py`` -> implements ASR 

The results pertaining to the answers of the rebuttal are organized in folders as follows:

1) **AutoPGD** -> transferability tables for pretrained and Robustbench models, for AutoPGD on CIFAR100, analogous to Tables 5–7 in Appendix B.

2) **Defenses** -> updated transferability Tables 3 and 5 for pretrained and Robustbench models, for the proposed attack, on CIFAR10/100, including the MeanSparse and MixedNuts defenses.

3) **Transformers** -> updated transferability Tables 3 and 5 for pretrained and Robustbench models, for the proposed attack, on CIFAR10/100, including Transformer-based architectures.

4) **Baselines** -> transferability tables for pretrained and Robustbench models, for the baseline attacks of the standard PGD and the F-PGD, on CIFAR10.
