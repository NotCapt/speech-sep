# GenCorr-Sep + ESSD (5-Speaker Speech Separation)
## 🧠 Architecture Overview
This model tries to solve the massive memory explosion of 5-speaker separation by using a novel **Early Split Shared Decoder (ESSD)**, and then overcomes the muffled "regression-to-the-mean" problem by using a secondary **Fast-GeCo flow-matching generative U-Net** to synthesize crisp acoustic details.

1. **Stage 1 (ESSD):** A memory-efficient Siamese Discriminative Separator. It splits the latent features early and processes all 5 speakers in parallel using shared weights, outputting coarse audio estimations.
2. **Stage 2 (Generative Corrector):** A Fast-GeCo Flow Matching U-Net that takes the coarse outputs from Stage 1 and refines them using an Euler ODE solver, restoring high-frequency details.

### Deep Technical Dive

**1. Stage 1: Early Split Shared Decoder (ESSD)**
Stage 1 is a purely discriminative model responsible for the heavy lifting of separating the speakers. Because separating 5 speakers simultaneously usually causes a combinatorial explosion in GPU memory, ESSD uses a strict dimensional bottleneck strategy:
* **Audio Encoder:** A 1D Convolution processes the raw waveform `[Batch, 1, Time]` into a high-dimensional, downsampled latent space `[Batch, Channels, Length]`.
* **Separation Encoder:** A stack of deep Transformer blocks processes this latent space to extract overlapping vocal patterns, saving "skip connections" at every layer.
* **The "Early Split" Layer:** Instead of maintaining a massive unified tensor, a 1x1 Convolution expands the channels and immediately reshapes the tensor into 5 distinct speaker streams: `[Batch, 5, Channels, Length]`. 
* **The Siamese Decoder:** This is the memory-saving core. We flatten the batch and speaker dimensions into `[Batch * 5, Channels, Length]`. By doing this, we pass all 5 speakers through a **single set of shared Transformer weights** in parallel. This completely eliminates the need for 5 independent decoders.
* **PIT Loss:** Because the model doesn't know *which* output channel corresponds to *which* speaker, we use the Hungarian Matching Algorithm to dynamically align the 5 predictions with the 5 ground truths to maximize the **SI-SNR (Scale-Invariant Signal-to-Noise Ratio)**.

**2. Stage 2: Generative Corrector (Fast-GeCo)**
By the time the audio leaves Stage 1, the 5 speakers are successfully separated, but they sound muffled and robotic. This is known as *regression-to-the-mean*—the model minimized its loss by predicting "safe", mathematically average frequencies. Stage 2 fixes this using Generative AI.
* **Flow Matching Framework:** Instead of using standard Diffusion (which takes 100+ steps and is too slow for audio), we use Flow Matching. Flow Matching learns a straight-line vector field from a starting point (x_0) to a target point (x_1).
* **The U-Net Score Network:** A massive 1D Generative U-Net. During training, it is fed the muffled audio from Stage 1, the original audio mixture (for conditioning), and a time embedding `t`. It learns to predict the "velocity" vector needed to push the muffled audio toward the pristine ground-truth audio.
* **The Euler ODE Solver:** During inference, we take the muffled Stage 1 output and set it as `t=0`. We then use a 4-step Euler ODE solver. At each of the 4 steps, the U-Net predicts the trajectory velocity, and we physically step the audio tensor forward. By `t=1`, the audio has been mathematically morphed into crisp, high-frequency speech!

## 📂 Repository Contents
* `speech-sep.ipynb`: The Jupyter Notebook containing the entire pipeline. It includes the datasets, dataloaders, neural network classes, training loops, and inference logic.

## 🚀 How to Run on Kaggle (Recommended)
This architecture is computationally heavy and is designed to be trained on high-end GPUs like the RTX 6000 or A100.

1. **Upload the Notebook:** Upload `speech-sep.ipynb` to Kaggle.
2. **Attach Data:** Attach the `Libri5Mix` dataset to your Kaggle notebook.
3. **Configure Paths:** Ensure `TRAIN_DATA_PATH` and `TEST_DATA_PATH` at the top of the notebook point to the correct Kaggle input directories.
4. **Train:** 
   * This will take up approximately 9.5 hrs
5. **Download Weights:** Once finished, navigate to the "Versions" tab of your notebook, go to the "Output" section, and download `stage1_essd_weights.pth` and `stage2_scorenet_weights.pth`.


## 📊 Evaluation
The model evaluates its separation performance mathematically using **Scale-Invariant Signal-to-Noise Ratio (SI-SNR)** via Permutation Invariant Training (PIT) matching.

## 📈 Training Status and Results
Please note that the visualizations, output graphs, and SI-SNR metrics currently present in the `speech-sep.ipynb` notebook are from an initial **30-epoch** training run. 

A full **90-epoch** version of this model is currently training on Kaggle. Because 5-speaker separation is highly complex, the generative refinement process benefits massively from longer training. The 90-epoch model is expected to produce significantly higher SI-SNR scores and much crisper acoustic detail once it finishes.
