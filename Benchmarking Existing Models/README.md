# Benchmarking Existing Models

This directory contains Jupyter notebooks used to benchmark state-of-the-art speech separation models. The evaluation focuses on separating multiple overlapping speakers from mixtures. 

## Models Evaluated

The following models are evaluated in the provided notebooks:

1. **SepFormer and ConvTasNet** (`sepformer-and-convtasnet.ipynb`)
   - **Dataset:** Libri3Mix-2 (3-speaker separation)
   - Benchmarks Attention-based Transformer (SepFormer) and Time-domain baseline (ConvTasNet).
2. **MossFormer2** (`mossformer2.ipynb`)
   - **Dataset:** Libri5Mix (5-speaker separation, wav16k/min)
   - Implements a hybrid Transformer and RNN-free Recurrent Network for 5-speaker separation (based on arXiv: 2312.11825).
3. **SVoice** (`svoice.ipynb`)
   - **Dataset:** Libri5Mix (5-speaker separation)

---

## Results and Inference

Below is a summary of the inferred evaluation metrics on the test sets from each notebook. 

### 1. 3-Speaker Separation (Libri3Mix-2)

Results extracted from `sepformer-and-convtasnet.ipynb`:

| Architecture | SI-SNRi (dB) | SI-SDRi (dB) | SDRi (dB) |
|--------------|--------------|--------------|-----------|
| **SepFormer**| 18.60        | 18.60        | 18.43     |
| **ConvTasNet**| 14.62        | 14.62        | 14.41     |

*Note: SepFormer demonstrates significantly higher performance on the 3-speaker task.*

### 2. 5-Speaker Separation (Libri5Mix)

Results extracted from `mossformer2.ipynb` and `svoice.ipynb`:

| Architecture | Average SI-SNR (dB) | SI-SNRi (dB) |
|--------------|---------------------|--------------|
| **MossFormer2** | -2.31               | 4.29         |
| **SVoice**      | -4.38               | 2.21         |

*Note: MossFormer2 provides an absolute improvement of ~2.08 dB in SI-SNRi over SVoice on the highly challenging 5-speaker separation task.*
