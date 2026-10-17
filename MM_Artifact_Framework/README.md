# MRI Motion Artifact Removal Framework 🧠🧲

## 1. Project Title & Abstract
**MRI Motion Artifacts Removal Model: A Two-Stage Artifact-Guided Reconstruction Network**

Motion artifacts, ghosting, and susceptibility issues frequently degrade the quality of Magnetic Resonance Imaging (MRI) scans, posing significant challenges for accurate clinical diagnosis. This project introduces a state-of-the-art, two-stage deep learning framework designed to isolate and adaptively remove motion artifacts from corrupted MRI volumes. By predicting an explicit artifact map and leveraging an innovative **3D Artifact-Aware Coordinate Attention** mechanism combined with **Pure Residual Correction**, the model meticulously preserves high-frequency anatomical structures while completely eliminating motion-induced smearing, thereby significantly improving diagnostic reliability.

## 2. Key Features
- **Two-Stage Architecture:** Explicit artifact prediction (Stage 1) followed by artifact-guided structural reconstruction (Stage 2).
- **Novel Attention Mechanism:** Integrates a custom 3D Artifact-Aware Coordinate Attention module to learn spatial artifact distributions and prioritize uncorrupted channels.
- **Pure Residual Correction:** Stage 2 explicitly predicts a structural correction map which is directly added to the input ($Final = HM2 + Correction$).
- **Blind Inference:** The framework is strictly designed to operate without the Clean Ground Truth MRI during inference, ensuring true real-world applicability.
- **Detail-Preserving Loss Formulation:** Joint optimization utilizing Charbonnier Loss, 3D Structural Similarity Index (SSIM), Focal Frequency Loss (FFL), 3D Spatial Gradient Loss, and an Artifact-Aware Preservation Loss to prevent oversmoothing and ensure the model only modifies corrupted tissue.

## 3. Data Preprocessing & Patch Generation
The framework handles raw NIfTI volumes strictly at their native dimension `(X, Y, Z) = (192, 256, 256)`. No artificial resizing or padding is applied.

### Step 3.1: Shared Subject-Level Normalization
Instead of normalizing independent 3D patches (which destroys global contrast relativity), we use **Shared Volume-Level Normalization**.
1. We compute robust minimum and maximum intensities: the $0.5^{th}$ and $99.5^{th}$ percentiles exclusively from the Clean volume.
2. These exact scalar values are applied universally to the Clean, HM1, and HM2 volumes for that subject, bounding them strictly between `[0.0, 1.0]`.

### Step 3.2: Patch Generation
1. We split the volume **only** along the Z-axis (Depth dimension).
2. The 256 depth slices are divided into exactly **16 non-overlapping patches**.
3. Each generated patch has the shape `(192, 256, 16)`.
4. These patches are saved as `.npy` arrays for rapid dataloading.

### Step 3.3: How to Preprocess from Scratch
If you are starting from scratch and do not have the preprocessed `.npy` files:
1. **Download the Dataset:** We use the [OpenNeuro ds004173 dataset](https://openneuro.org/datasets/ds004173) which contains standard T1w (`clean`), and motion-corrupted T1w (`hm1`, `hm2`) scans.
2. **Configure Paths:** Open `scripts/preprocess_data.py` and modify the `src_dir` to point to your downloaded raw dataset, and `dest_dir` to your desired output folder.
3. **Run Preprocessing:**
   ```bash
   python scripts/preprocess_data.py
   ```
4. **Data Structure & Split:** The script automatically discovers all subjects in your `src_dir` (e.g., all 148 subjects) and performs an automatic 80/20 train/val split. It creates `data/train` and `data/val` folders inside your `dest_dir`, containing the normalized `.npy` patches. 
5. **Upload to Kaggle/Cloud:** If you are training on a cloud GPU (like Kaggle), zip and upload this generated `dest_dir` as your training dataset. In `configs/config.py`, make sure `DATA_DIR` points to this uploaded dataset path.

## 4. Internal Model Architecture & Detailed Pipeline

The framework is divided into two explicitly defined stages. Here is a detailed breakdown of how each stage and its internal components perform their tasks:

### Stage 1: Artifact Map Predictor
Stage 1 is designed to predict the structural differences (Artifact Map) between the corrupted images and the clean targets.
- **Structure (3D U-Net Variant):** It consists of a 3D Encoder-Decoder architecture.
  - **Encoder Pipeline:** The encoder takes the concatenated corrupted inputs (HM1 and HM2) and progressively downsamples them spatially (while maintaining depth). Its main task is to extract high-level feature representations and understand the spatial distribution and severity of the motion artifacts.
  - **Decoder Pipeline:** The decoder takes these compressed features and progressively upsamples them back to the original resolution using skip connections from the encoder. Its task is to accurately reconstruct the precise spatial locations and intensities of the artifacts, outputting the final artifact map.
- **Output:** The predicted artifact map $\hat{A}$, which acts as a spatial and semantic guide for Stage 2.

### Stage 2: Artifact-Guided Reconstructor
Stage 2 utilizes the artifact map from Stage 1 to adaptively reconstruct the clean MRI without ever seeing the ground truth clean MRI during inference.

#### 1. Dual Encoders
The network employs two separate 3D Encoders to process different types of information:
- **HM2 Encoder ($E_{HM2}$):** A standard 3D Encoder that extracts rich **structural features** ($F_{HM2}$) from the corrupted HM2 patch. It maintains the hierarchical structural integrity of the brain anatomy.
- **Artifact Encoder ($E_{art}$):** A Lightweight 3D Encoder that extracts **semantic artifact features** ($F_{art}$) from the predicted artifact map. Since the artifact map contains less structural complexity than the raw MRI, this encoder is purposefully lightweight to prevent overfitting.

#### 2. Feature Fusion (How and What it Extracts)
Instead of simply concatenating and passing features blindly, the framework employs an explicit **Feature Fusion Module** at the bottleneck.
- **What it extracts/does:** It intelligently blends the raw anatomical structural features from the HM2 encoder with the spatial artifact guidance features from the Artifact encoder.
- **How it works:** It concatenates the two feature tensors along the channel dimension and immediately applies a **1x1x1 Convolution** (`nn.Conv3d(kernel_size=1)`). This cross-channel compression allows the network to learn exactly which structural features are corrupted (guided by the artifact features) and seamlessly integrates them into a unified, fused feature representation before attention is applied.

#### 3. Artifact-Aware Coordinate Attention (3D)
To adaptively suppress artifact-corrupted regions, we extend Coordinate Attention to 3D. This module takes the fused features and learns spatial and channel-wise relationships, actively prioritizing uncorrupted channels and suppressing regions heavily affected by motion.

#### 4. Residual Refinement & Decoder
- **Residual Blocks:** The attended features pass through a series of 3D Residual Blocks to refine the representations and stabilize training.
- **Decoder Pipeline:** The 3D Decoder gradually upsamples the refined features back to the native spatial resolution. It utilizes hierarchical skip connections (`f_high`, `f_mid`) from the HM2 Encoder to explicitly reintroduce high-frequency anatomical details (like edges and fine tissues) that were lost during the encoder's downsampling process.

#### 5. Pure Residual Correction
Instead of outputting the final clean MRI directly, the Stage 2 Decoder outputs a **Correction Map** ($C_{pred}$). 
- **Reconstruction Equation**: The final volume is obtained by adding the predicted correction directly to the corrupted input, ensuring only corrupted tissues are modified:
  $$ \hat{Clean} = HM2 + C_{pred} $$

## 5. Loss Function Mathematical Formulations

The multi-loss framework has been strictly optimized to prevent oversmoothing and ensure edges are preserved across all spatial dimensions.

### Stage 1: Artifact Loss (L1)
Supervises the predicted artifact map to match the absolute difference between the corrupted HM2 and Clean scans.
- **Formula:** $\mathcal{L}_{art} = \frac{1}{N} \sum | \hat{A} - (HM2 - Clean) |$

### Stage 2: Charbonnier Loss (Weight: 1.0)
A robust version of L1 loss that prevents artifacts generated by standard L1/MSE by using a small epsilon for differentiability at zero.
- **Formula:** $\mathcal{L}_{char} = 	ext{mean}(\sqrt{(\hat{Clean} - Clean)^2 + \epsilon^2})$

### Stage 2: 3D Structural Similarity Index (SSIM) (Weight: 0.5)
Focuses on local structural integrity, luminance, and contrast using a 3D Gaussian window.
- **Formula:** $\mathcal{L}_{SSIM} = 1 - 	ext{SSIM}(\hat{Clean}, Clean)$

### Stage 2: True 3D Focal Frequency Loss (Weight: 0.1 - 1.0)
Operates on the full 3D volumetric frequency spectrum across depth, height, and width. It dynamically downweights easily reconstructed low frequencies and focuses the network on harder-to-learn spatial frequencies where 3D motion artifacts typically reside.
- **Formula:** $\mathcal{L}_{FFL3D} = \text{mean}( W(d, h, w) \times || \mathcal{F}_{3D}(\hat{Clean}) - \mathcal{F}_{3D}(Clean) ||^2 )$

### Stage 2: 3D Spatial Gradient Loss (Weight: 1.0)
Calculates the L1 difference between the spatial gradients across all 3 dimensions (Depth, Height, Width) using explicit tensor shifts to preserve sharp anatomical edges.
- **Formula:** $\mathcal{L}_{grad} = || \nabla_D \hat{Clean} - \nabla_D Clean ||_1 + || \nabla_H \hat{Clean} - \nabla_H Clean ||_1 + || \nabla_W \hat{Clean} - \nabla_W Clean ||_1$

### Stage 2: Artifact-Aware Preservation Loss (Weight: 2.0)
A dynamic weight mask created during training that exponentially penalizes the `PredictedCorrection` magnitude if the region is already aligned with the `Clean` target. This prevents the model from unnecessarily smoothing or modifying pristine tissue.
- **Formula:** $\mathcal{L}_{preserve} = 	ext{mean}( e^{-50 \times |HM2 - Clean|} \times |C_{pred}| )$

*(Note: Perceptual and explicit Edge losses were tested and retired in favor of the mathematically robust 3D Gradient and Preservation losses above).*

### Stage 2: Total Loss Calculation
The final objective function for Stage 2 is a weighted sum of all the individual losses.
- **Formula:** $\mathcal{L}_{total} = (\lambda_{char} \times \mathcal{L}_{char}) + (\lambda_{SSIM} \times \mathcal{L}_{SSIM}) + (\lambda_{FFL} \times \mathcal{L}_{FFL}) + (\lambda_{grad} \times \mathcal{L}_{grad}) + (\lambda_{preserve} \times \mathcal{L}_{preserve})$
- **Current Configured Weights:** 
  $$\mathcal{L}_{total} = (1.0 \times \mathcal{L}_{char}) + (0.5 \times \mathcal{L}_{SSIM}) + (0.1 \times \mathcal{L}_{FFL}) + (1.0 \times \mathcal{L}_{grad}) + (2.0 \times \mathcal{L}_{preserve})$$

## 6. Normalization & Metric Evaluation (PSNR/SSIM)

A critical aspect of our framework is how normalization is handled during input preprocessing and metric evaluation:

- **Inputs (Clean, HM1, HM2):** As detailed in Section 3.1, all input volumes are normalized strictly to `[0.0, 1.0]` using shared volume-level robust percentiles ($0.5^{th}$ and $99.5^{th}$ of the Clean volume).
- **Stage 1 (Predicted Artifact Map) & Stage 2 (Reconstructed Output):** The predicted artifact map and the final reconstructed output (`HM2 + Correction`) are **NOT** normalized, scaled, or passed through any bounding activations (like Sigmoid or Tanh) before calculating PSNR and SSIM.

**Technical Reason for Unbounded Outputs vs Normalized Targets:**
You might wonder: *How can we calculate PSNR/SSIM between an un-normalized predicted output and a strictly normalized Clean MRI?*

The answer lies in the training objective and the implicit domain mapping:
1. **Shared Domain Constraint:** The input (`HM2`) is already normalized to `[0.0, 1.0]`. The network predicts a `Correction` map, and the final output is `HM2 + Correction`. Because the loss functions strictly penalize the difference between this output and the `[0.0, 1.0]` normalized Clean MRI, the network *implicitly learns* to constrain its final output to the exact same `[0.0, 1.0]` domain.
2. **Exposing Residual Errors:** By not artificially forcing the final output through a clipping function (like `torch.clamp` or `Sigmoid`), we leave the network vulnerable to making out-of-bounds errors (e.g., predicting `1.2` or `-0.1`). Comparing these raw, unbounded predictions directly against the normalized `[0.0, 1.0]` target accurately penalizes the model for overshooting and correctly lowers the PSNR/SSIM. If we were to clip the output to `[0.0, 1.0]` before calculating metrics, we would hide these predictive failures and falsely inflate the scores.
3. **Pure Residual Learning:** Allowing the network to naturally converge without artificial activation boundaries forces it to learn the precise mathematical correction map rather than lazily relying on the saturation limits of a Sigmoid function. This leads to mathematically robust convergence and better clinical generalization.
## 7. Training & Inference Pipeline Usage

### Inference Pipeline (Evaluation Modes)
The test script is designed to handle normalization and processing completely autonomously. It accepts raw `.npy` volumes, applies rigorous global percentile normalization on the full volumes first, and strictly avoids any patch-level normalization.

#### 1. Full-Volume Mode (`--eval_mode full_volume`)
- The script evaluates the entire normalized 3D volume through Stage-1 and Stage-2 in a single pass.

#### 2. Patch-Based Mode (`--eval_mode patch_based`)
- The script pre-normalizes the full volume, then splits it into exactly **16 non-overlapping patches** (`192 x 256 x 16`).
- Each patch is evaluated independently to simulate training-style inputs.
- The patches are then assembled back into a RAW full volume before a single, final global normalization is applied for metric evaluation.

## 8. Step-by-Step Training & Inference Guide

### Step 1: Environment Setup
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/coderisalive/agcnet2-model.git
cd agcnet2-model
pip install -r requirements.txt
```

### Step 2: Data Seeding & Preprocessing
To train from scratch, you must first preprocess your raw NIfTI data into normalized patches.
1. Download the [OpenNeuro ds004173 dataset](https://openneuro.org/datasets/ds004173).
2. Open `scripts/preprocess_data.py` and set `src_dir` to your downloaded dataset and `dest_dir` to your output folder.
3. Run the preprocessing script (this automatically discovers all subjects and splits them exactly 50/30/20 for train/val/test):
```bash
python scripts/preprocess_data.py
```
4. If using Kaggle/Cloud, zip the `dest_dir` and upload it. Set `DATA_DIR` in `configs/config.py` to point to this location.

### Step 3: Automated End-to-End Training (Stage 1 & Stage 2)
We have introduced a fully automated training pipeline (`--stage 3`). 
This command will automatically:
1. Run **Stage 1** (Artifact Predictor) to completion.
2. Grab the best checkpoint from Stage 1.
3. Automatically initialize and start **Stage 2** (Artifact-Guided Reconstructor) without any manual intervention.

```bash
python scripts/train.py --stage 3 --epochs 50 --batch_size 2
```

### Step 4: How and Where Are Checkpoints Saved?
All checkpoints are automatically routed to a centralized folder defined in `configs/config.py` (`C:\Users\bitpa\OneDrive\Desktop\result`). Both stages save their files in **completely separate folders** so they never mix.

Here is the exact folder structure and saving behavior:
```text
C:\Users\bitpa\OneDrive\Desktop\result
│
├── stage1\
│   └── model_best_loss.pth      <-- (Stage 1 keeps the final best model after training completes)
│
└── stage2_final_detail_preserved\
    ├── model_best_loss.pth      <-- (Best Stage 2 model overall)
    ├── epoch5_checkpoint.pth    <-- (Saved at 5th epoch, never overwritten)
    ├── epoch10_checkpoint.pth   <-- (Saved at 10th epoch, never overwritten)
    └── ... (Saves every 5 epochs)
```
- **Stage 1 Behavior**: It will train normally and at the end of the entire training, it passes the best model's path (`stage1\model_best_loss.pth`) directly to Stage 2.
- **Stage 2 Custom Interval Saving**: To prevent losing your training progress, Stage 2 creates a brand new snapshot file every **5 epochs**. This ensures earlier epochs are never overwritten.

### Step 5: Run Inference / Testing
Test your fully trained pipeline. Our testing script supports multi-checkpoint evaluation, meaning you can test as many Stage-2 checkpoints as you want simultaneously against a single cached Stage-1 execution.

#### Quick Evaluation (Best Models Only)
To quickly test the overall best models from both Stage 1 and Stage 2 (for example, if your checkpoints are located in `E:/myenv/MRI-checkpoints`):

```bash
python scripts/test.py \
    --stage1_checkpoint "E:/myenv/MRI-checkpoints/stage1/model_best_loss.pth" \
    --stage2_checkpoints "E:/myenv/MRI-checkpoints/stage2_final_detail_preserved/model_best_loss.pth" \
    --eval_mode patch_based
```

#### Option A: Patch-Based Evaluation (Simulates Training)
To test the best model alongside the checkpoints saved every 5 epochs in Patch-Based mode:

```bash
python scripts/test.py \
    --eval_mode patch_based \
    --stage1_checkpoint C:\Users\bitpa\OneDrive\Desktop\result\stage1\model_best_loss.pth \
    --stage2_checkpoints \
        C:\Users\bitpa\OneDrive\Desktop\result\stage2_final_detail_preserved\model_best_loss.pth \
        C:\Users\bitpa\OneDrive\Desktop\result\stage2_final_detail_preserved\epoch5_checkpoint.pth \
        C:\Users\bitpa\OneDrive\Desktop\result\stage2_final_detail_preserved\epoch10_checkpoint.pth \
        C:\Users\bitpa\OneDrive\Desktop\result\stage2_final_detail_preserved\epoch15_checkpoint.pth \
        C:\Users\bitpa\OneDrive\Desktop\result\stage2_final_detail_preserved\epoch20_checkpoint.pth
```

#### Option B: Full-Volume Evaluation (Single Forward Pass)
If you want to evaluate the entire 3D volume at once (requires more VRAM):

```bash
python scripts/test.py \
    --eval_mode full_volume \
    --stage1_checkpoint C:\Users\bitpa\OneDrive\Desktop\result\stage1\model_best_loss.pth \
    --stage2_checkpoints \
If you are starting from scratch and do not have the preprocessed `.npy` files:
1. **Download the Dataset:** We use the [OpenNeuro ds004173 dataset](https://openneuro.org/datasets/ds004173) which contains standard T1w (`clean`), and motion-corrupted T1w (`hm1`, `hm2`) scans.
2. **Configure Paths:** Open `scripts/preprocess_data.py` and modify the `src_dir` to point to your downloaded raw dataset, and `dest_dir` to your desired output folder.
3. **Run Preprocessing:**
   ```bash
   python scripts/preprocess_data.py
   ```
4. **Data Structure & Split:** The script automatically discovers all subjects in your `src_dir` (e.g., all 148 subjects) and performs an automatic 80/20 train/val split. It creates `data/train` and `data/val` folders inside your `dest_dir`, containing the normalized `.npy` patches. 
5. **Upload to Kaggle/Cloud:** If you are training on a cloud GPU (like Kaggle), zip and upload this generated `dest_dir` as your training dataset. In `configs/config.py`, make sure `DATA_DIR` points to this uploaded dataset path.

## 4. Internal Model Architecture & Detailed Pipeline

The framework is divided into two explicitly defined stages. Here is a detailed breakdown of how each stage and its internal components perform their tasks:

### Stage 1: Artifact Map Predictor
Stage 1 is designed to predict the structural differences (Artifact Map) between the corrupted images and the clean targets.
- **Structure (3D U-Net Variant):** It consists of a fully learnable 3D Encoder-Decoder architecture. All non-learnable pooling and scaling operations (`MaxPool3d`, `F.interpolate`) have been explicitly removed and replaced with spatial strided convolutions (`Conv3d`, stride=2) and transposed convolutions (`ConvTranspose3d`, stride=2).
  - **Encoder Pipeline:** The encoder takes the concatenated corrupted inputs (HM1 and HM2) and progressively downsamples them spatially. Crucially, the depth dimension is never downsampled (Stride Z=1), ensuring the 16 volume slices are perfectly preserved across all layers.
  - **Decoder Pipeline:** The decoder takes these compressed features and progressively upsamples them back to the exact original resolution using learnable transposed convolutions and skip connections from the encoder.
- **Input / Output Dimensionality:** If a corrupted patch of size `16 × 192 × 256` (D x H x W) is fed as input, the network strictly maintains spatial integrity and outputs a predicted artifact map of exactly `16 × 192 × 256`.
- **Mathematical Representation:** The network explicitly predicts the artifact map $\hat{A}$ which approximates the residual difference between the corrupted scan and the clean target:
  $$ \hat{A} = E_{stage1}(HM1, HM2) \approx (HM2 - Clean) $$

### Stage 2: Artifact-Guided Reconstructor
Stage 2 utilizes the artifact map from Stage 1 to adaptively reconstruct the clean MRI without ever seeing the ground truth clean MRI during inference.

#### 1. Dual Encoders
The network employs two separate 3D Encoders to process different types of information:
- **HM2 Encoder ($E_{HM2}$):** A standard 3D Encoder that extracts rich **structural features** ($F_{HM2}$) from the corrupted HM2 patch. It maintains the hierarchical structural integrity of the brain anatomy.
- **Artifact Encoder ($E_{art}$):** A Lightweight 3D Encoder that extracts **semantic artifact features** ($F_{art}$) from the predicted artifact map. Since the artifact map contains less structural complexity than the raw MRI, this encoder is purposefully lightweight to prevent overfitting.

#### 2. Feature Fusion (How and What it Extracts)
Instead of simply concatenating and passing features blindly, the framework employs an explicit **Feature Fusion Module** at the bottleneck.
- **What it extracts/does:** It intelligently blends the raw anatomical structural features from the HM2 encoder with the spatial artifact guidance features from the Artifact encoder.
- **How it works:** It concatenates the two feature tensors along the channel dimension and immediately applies a **1x1x1 Convolution** (`nn.Conv3d(kernel_size=1)`). This cross-channel compression allows the network to learn exactly which structural features are corrupted (guided by the artifact features) and seamlessly integrates them into a unified, fused feature representation before attention is applied.

#### 3. Artifact-Aware Coordinate Attention (3D)
To adaptively suppress artifact-corrupted regions, we extend Coordinate Attention to 3D. This module takes the fused features and learns spatial and channel-wise relationships, actively prioritizing uncorrupted channels and suppressing regions heavily affected by motion.

#### 4. Residual Refinement & Decoder
- **Residual Blocks:** The attended features pass through a series of 3D Residual Blocks to refine the representations and stabilize training.
- **Decoder Pipeline:** The 3D Decoder gradually upsamples the refined features back to the native spatial resolution using learnable **Transposed Convolutions** (`ConvTranspose3d`, stride=2) rather than static interpolation. It utilizes hierarchical skip connections (`f_high`, `f_mid`) from the HM2 Encoder to explicitly reintroduce high-frequency anatomical details (like edges and fine tissues) that were lost during the encoder's downsampling process.

#### 5. Pure Residual Correction
Instead of outputting the final clean MRI directly, the Stage 2 Decoder outputs a **Correction Map** ($C_{pred}$). 
- **Reconstruction Equation**: The final volume is obtained by adding the predicted correction directly to the corrupted input, ensuring only corrupted tissues are modified:
  $$ \hat{Clean} = HM2 + C_{pred} $$

## 5. Loss Function Mathematical Formulations

The multi-loss framework has been strictly optimized to prevent oversmoothing and ensure edges are preserved across all spatial dimensions.

### Stage 1: Artifact Loss (L1)
Supervises the predicted artifact map to match the absolute difference between the corrupted HM2 and Clean scans.
- **Formula:** $\mathcal{L}_{art} = \frac{1}{N} \sum | \hat{A} - (HM2 - Clean) |$

### Stage 2: Charbonnier Loss (Weight: 1.0)
A robust version of L1 loss that prevents artifacts generated by standard L1/MSE by using a small epsilon for differentiability at zero.
- **Formula:** $\mathcal{L}_{char} = 	ext{mean}(\sqrt{(\hat{Clean} - Clean)^2 + \epsilon^2})$

### Stage 2: 3D Structural Similarity Index (SSIM) (Weight: 0.5)
Focuses on local structural integrity, luminance, and contrast using a 3D Gaussian window.
- **Formula:** $\mathcal{L}_{SSIM} = 1 - 	ext{SSIM}(\hat{Clean}, Clean)$

### Stage 2: True 3D Focal Frequency Loss (Weight: 0.1 - 1.0)
Operates on the full 3D volumetric frequency spectrum across depth, height, and width. It dynamically downweights easily reconstructed low frequencies and focuses the network on harder-to-learn spatial frequencies where 3D motion artifacts typically reside.
- **Formula:** $\mathcal{L}_{FFL3D} = \text{mean}( W(d, h, w) \times || \mathcal{F}_{3D}(\hat{Clean}) - \mathcal{F}_{3D}(Clean) ||^2 )$

### Stage 2: 3D Spatial Gradient Loss (Weight: 1.0)
Calculates the L1 difference between the spatial gradients across all 3 dimensions (Depth, Height, Width) using explicit tensor shifts to preserve sharp anatomical edges.
- **Formula:** $\mathcal{L}_{grad} = || \nabla_D \hat{Clean} - \nabla_D Clean ||_1 + || \nabla_H \hat{Clean} - \nabla_H Clean ||_1 + || \nabla_W \hat{Clean} - \nabla_W Clean ||_1$

### Stage 2: Artifact-Aware Preservation Loss (Weight: 2.0)
A dynamic weight mask created during training that exponentially penalizes the `PredictedCorrection` magnitude if the region is already aligned with the `Clean` target. This prevents the model from unnecessarily smoothing or modifying pristine tissue.
- **Formula:** $\mathcal{L}_{preserve} = 	ext{mean}( e^{-50 \times |HM2 - Clean|} \times |C_{pred}| )$

*(Note: Perceptual and explicit Edge losses were tested and retired in favor of the mathematically robust 3D Gradient and Preservation losses above).*

### Stage 2: Total Loss Calculation
The final objective function for Stage 2 is a weighted sum of all the individual losses.
- **Formula:** $\mathcal{L}_{total} = (\lambda_{char} \times \mathcal{L}_{char}) + (\lambda_{SSIM} \times \mathcal{L}_{SSIM}) + (\lambda_{FFL} \times \mathcal{L}_{FFL}) + (\lambda_{grad} \times \mathcal{L}_{grad}) + (\lambda_{preserve} \times \mathcal{L}_{preserve})$
- **Current Configured Weights:** 
  $$\mathcal{L}_{total} = (1.0 \times \mathcal{L}_{char}) + (0.5 \times \mathcal{L}_{SSIM}) + (0.1 \times \mathcal{L}_{FFL}) + (1.0 \times \mathcal{L}_{grad}) + (2.0 \times \mathcal{L}_{preserve})$$

## 6. Normalization & Metric Evaluation (PSNR/SSIM)

A critical aspect of our framework is how normalization is handled during input preprocessing and metric evaluation:

- **Inputs (Clean, HM1, HM2):** As detailed in Section 3.1, all input volumes are normalized strictly to `[0.0, 1.0]` using shared volume-level robust percentiles ($0.5^{th}$ and $99.5^{th}$ of the Clean volume).
- **Stage 1 (Predicted Artifact Map) & Stage 2 (Reconstructed Output):** The predicted artifact map and the final reconstructed output (`HM2 + Correction`) are **NOT** normalized, scaled, or passed through any bounding activations (like Sigmoid or Tanh) before calculating PSNR and SSIM.

**Technical Reason for Unbounded Outputs vs Normalized Targets:**
You might wonder: *How can we calculate PSNR/SSIM between an un-normalized predicted output and a strictly normalized Clean MRI?*

The answer lies in the training objective and the implicit domain mapping:
1. **Shared Domain Constraint:** The input (`HM2`) is already normalized to `[0.0, 1.0]`. The network predicts a `Correction` map, and the final output is `HM2 + Correction`. Because the loss functions strictly penalize the difference between this output and the `[0.0, 1.0]` normalized Clean MRI, the network *implicitly learns* to constrain its final output to the exact same `[0.0, 1.0]` domain.
2. **Exposing Residual Errors:** By not artificially forcing the final output through a clipping function (like `torch.clamp` or `Sigmoid`), we leave the network vulnerable to making out-of-bounds errors (e.g., predicting `1.2` or `-0.1`). Comparing these raw, unbounded predictions directly against the normalized `[0.0, 1.0]` target accurately penalizes the model for overshooting and correctly lowers the PSNR/SSIM. If we were to clip the output to `[0.0, 1.0]` before calculating metrics, we would hide these predictive failures and falsely inflate the scores.
3. **Pure Residual Learning:** Allowing the network to naturally converge without artificial activation boundaries forces it to learn the precise mathematical correction map rather than lazily relying on the saturation limits of a Sigmoid function. This leads to mathematically robust convergence and better clinical generalization.
## 7. Training & Inference Pipeline Usage

### Inference Pipeline (Evaluation Modes)
The test script is designed to handle normalization and processing completely autonomously. It accepts raw `.npy` volumes, applies rigorous global percentile normalization on the full volumes first, and strictly avoids any patch-level normalization.

#### 1. Full-Volume Mode (`--eval_mode full_volume`)
- The script evaluates the entire normalized 3D volume through Stage-1 and Stage-2 in a single pass.

#### 2. Patch-Based Mode (`--eval_mode patch_based`)
- The script pre-normalizes the full volume, then splits it into exactly **16 non-overlapping patches** (`192 x 256 x 16`).
- Each patch is evaluated independently to simulate training-style inputs.
- The patches are then assembled back into a RAW full volume before a single, final global normalization is applied for metric evaluation.

## 8. Step-by-Step Training & Inference Guide

### Step 1: Environment Setup
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/coderisalive/agcnet2-model.git
cd agcnet2-model
pip install -r requirements.txt
```

### Step 2: Data Seeding & Preprocessing
To train from scratch, you must first preprocess your raw NIfTI data into normalized patches.
1. Download the [OpenNeuro ds004173 dataset](https://openneuro.org/datasets/ds004173).
2. Open `scripts/preprocess_data.py` and set `src_dir` to your downloaded dataset and `dest_dir` to your output folder.
3. Run the preprocessing script (this automatically discovers all subjects and splits them exactly 50/30/20 for train/val/test):
```bash
python scripts/preprocess_data.py
```
4. If using Kaggle/Cloud, zip the `dest_dir` and upload it. Set `DATA_DIR` in `configs/config.py` to point to this location.

### Step 3: Automated End-to-End Training (Stage 1 & Stage 2)
We have introduced a fully automated training pipeline (`--stage 3`). 
This command will automatically:
1. Run **Stage 1** (Artifact Predictor) to completion.
2. Grab the best checkpoint from Stage 1.
3. Automatically initialize and start **Stage 2** (Artifact-Guided Reconstructor) without any manual intervention.

```bash
python scripts/train.py --stage 3 --epochs 50 --batch_size 2
```

### Step 4: How and Where Are Checkpoints Saved?
All checkpoints are automatically routed to a centralized folder defined in `configs/config.py` (`C:\Users\bitpa\OneDrive\Desktop\result`). Both stages save their files in **completely separate folders** so they never mix.

Here is the exact folder structure and saving behavior:
```text
C:\Users\bitpa\OneDrive\Desktop\result
│
├── stage1\
│   └── model_best_loss.pth      <-- (Stage 1 keeps the final best model after training completes)
│
└── stage2_final_detail_preserved\
    ├── model_best_loss.pth      <-- (Best Stage 2 model overall)
    ├── epoch5_checkpoint.pth    <-- (Saved at 5th epoch, never overwritten)
    ├── epoch10_checkpoint.pth   <-- (Saved at 10th epoch, never overwritten)
    └── ... (Saves every 5 epochs)
```
- **Stage 1 Behavior**: It will train normally and at the end of the entire training, it passes the best model's path (`stage1\model_best_loss.pth`) directly to Stage 2.
- **Stage 2 Custom Interval Saving**: To prevent losing your training progress, Stage 2 creates a brand new snapshot file every **5 epochs**. This ensures earlier epochs are never overwritten.

### Step 5: Run Inference / Testing
Test your fully trained pipeline. Our testing script supports multi-checkpoint evaluation, meaning you can test as many Stage-2 checkpoints as you want simultaneously against a single cached Stage-1 execution.

#### Quick Evaluation (Best Models Only)
To quickly test the overall best models from both Stage 1 and Stage 2 (for example, if your checkpoints are located in `E:/myenv/MRI-checkpoints`):

```bash
python scripts/test.py \
    --stage1_checkpoint "E:/myenv/MRI-checkpoints/stage1/model_best_loss.pth" \
    --stage2_checkpoints "E:/myenv/MRI-checkpoints/stage2_final_detail_preserved/model_best_loss.pth" \
    --eval_mode patch_based
```

#### Option A: Patch-Based Evaluation (Simulates Training)
To test the best model alongside the checkpoints saved every 5 epochs in Patch-Based mode:

```bash
python scripts/test.py \
    --eval_mode patch_based \
    --stage1_checkpoint C:\Users\bitpa\OneDrive\Desktop\result\stage1\model_best_loss.pth \
    --stage2_checkpoints \
        C:\Users\bitpa\OneDrive\Desktop\result\stage2_final_detail_preserved\model_best_loss.pth \
        C:\Users\bitpa\OneDrive\Desktop\result\stage2_final_detail_preserved\epoch5_checkpoint.pth \
        C:\Users\bitpa\OneDrive\Desktop\result\stage2_final_detail_preserved\epoch10_checkpoint.pth \
        C:\Users\bitpa\OneDrive\Desktop\result\stage2_final_detail_preserved\epoch15_checkpoint.pth \
        C:\Users\bitpa\OneDrive\Desktop\result\stage2_final_detail_preserved\epoch20_checkpoint.pth
```

#### Option B: Full-Volume Evaluation (Single Forward Pass)
If you want to evaluate the entire 3D volume at once (requires more VRAM):

```bash
python scripts/test.py \
    --eval_mode full_volume \
    --stage1_checkpoint C:\Users\bitpa\OneDrive\Desktop\result\stage1\model_best_loss.pth \
    --stage2_checkpoints \
        C:\Users\bitpa\OneDrive\Desktop\result\stage2_final_detail_preserved\model_best_loss.pth \
        C:\Users\bitpa\OneDrive\Desktop\result\stage2_final_detail_preserved\epoch5_checkpoint.pth \
        C:\Users\bitpa\OneDrive\Desktop\result\stage2_final_detail_preserved\epoch10_checkpoint.pth
```

#### Option C: Individual Patch Testing (Isolated Visualization)
If you want to test and visualize a specific patch (e.g., patch 8) independently without concatenating them into a full volume:

```bash
python scripts/test_single_patches.py \
    --stage1_checkpoint "E:/myenv/MRI-checkpoints/stage1/model_best_loss.pth" \
    --stage2_checkpoint "E:/myenv/MRI-checkpoints/stage2_final_detail_preserved/model_best_loss.pth" \
    --patches 8
```
*(Visualizations for individual patches will be saved in the `output_patches` directory).*
