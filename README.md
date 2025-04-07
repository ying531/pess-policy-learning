<h1 align="center">
<p> Reproduction code for Pessimistic Policy Learning
</h1> 

This repository contains code for reproducing results in the paper [Policy learning "without" overlap: Pessimism and generalized empirical Bernstein's inequality](https://arxiv.org/abs/2212.09900).


### Usage 
First, running `install.sh` installs necessary packages for the experiments, which executes `setup.py`.


Folder `experiments` contains scripts for reproducing the experiments:
- `MAB_batch.py`: experiments in Section 7.1.1 (PPL).
- `MAB_batch_clip.py`: experiments in Section 7.1.1 (with clipping).
- `synthetic_dt_linpess.py`: experiments in Section 7.1.2 (with linear PEVI).
- `synthetic_linear.py`: experiments in Section 7.2.1 (TS contextual bandit with well-specified exploration).
- `synthetic_opt.py`: experiments in Section 7.2.2 and 7.2.4 (cross validation) (TS contextual bandit with optimal overlap).
- `synthetic_miss.py`: experiments in Section 7.2.3 (TS contextual bandit with misspecified exploration).
- `real.py`: experiments in Section 7.3 (real datasets).
 
### Other files
Folder `utils` contains code for data generation and thopmson sampling. 

Folder `algs` contains the key algorithms for policy tree search, pessimistic policy learning, and cross validation.


#### Reference 

<a name="reference"></a>
```
@article{jin2025policy,
  title={Policy learning" without" overlap: Pessimism and generalized empirical bernstein's inequality},
  author={Jin, Ying and Ren, Zhimei and Yang, Zhuoran and Wang, Zhaoran},
  journal={Annals of Statistics (accepted)},
  year={2025+}
}
```