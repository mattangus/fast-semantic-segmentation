Train network from `README.md`.

Run in this order:
- `compute_stats.py`
  - writes mean and cov_inv to `<eval_dir>/stats/`
#- `ood.py` with `--write_out` and `--do_ood` options
#  - writes distances and logits to `<eval_dir>/dump_<type>_<eps>`
#- `normalise_data.py`
#  - writes mean and std to `<eval_dir>/dump_<type>_<eps>/normalisation.npy`
- re-run `ood.py` with `--do_ood` and `--use_train` options
  - compute roc curve for threshold
  - input threshold to `ood.py`
- re-run `ood.py` with `--do_ood` option
  - this will evaluate `mIoU` of threshold
