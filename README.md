# ASUNDER

Layout-aware cross-replica learning of common-mode fault vulnerability for directive and floorplan co-design of redundant HLS designs.

## Quick start

Open `ASUNDER_full_pipeline.ipynb` in Google Colab (free CPU runtime) or Jupyter and run all cells. The notebook writes every module it needs, so it also runs on its own. `FAST = True` uses a reduced configuration; `FAST = False` reproduces the settings of the paper.

## Running the scripts directly

All files sit in this one folder. Create the working folders first: `mkdir -p analysis figs data results models search graphs work`.

1. `python gen_data.py aes_ark,aes_mix,present,speck,crc16,fir4,dct4,cordic` builds the labelled dataset (HLS, Yosys synthesis, placement, fault injection).
2. `python cache_graphs.py aes_ark,aes_mix,present,speck,crc16,fir4,dct4,cordic` builds the cross-replica graphs.
3. `python exp_pred.py asunder:0:0 gcn:0:0 ...` trains models (`model:fold:seed`); `python gbm.py` runs the gradient-boosting baseline.
4. `python stats.py`, `python ablation.py`, `python riskmap.py` and `python e0.py` compute the statistics.
5. `python gen_transfer.py <kernels>` then `python transfer.py` evaluate threats never used in training.
6. Run `search.run_kernel(k)` for each kernel, then `python search_gp.py <kernels>` and `python search_stats.py`.
7. `fig_arch.py`, `fig_threat.py`, `fig_interaction.py`, `fig_results.py` and `fig_threats_ablation.py` draw the figures.

Environment variables `ASUNDER_N`, `ASUNDER_LANES`, `ASUNDER_NTRAIN`, `ASUNDER_EPOCHS`, `ASUNDER_POOL`, `ASUNDER_BUDGET` and `ASUNDER_SEARCH_SEEDS` set the scale. Yosys comes from pip as `yowasp-yosys`; no other EDA tool is needed.
