# README

---

## 1. Introduction / Background

Magnetic Resonance Imaging (MRI) reconstruction is a classic inverse problem where the goal is to recover a high-quality image $x$ from noisy, undersampled measurements $y$ acquired in $k$-space. The measurement process is modeled by a forward operator $A$:

$$
  y = A(x) + \text{noise}
$$

This captures the underlying physics of MRI sensing. Traditionally, MRI reconstruction pipelines are designed and tuned for specific anatomies—such as brain or knee imaging—which necessitates separate models and extensive parameter adjustments for each case.

Recent advances in self-supervised representation learning have provided compelling evidence for its potential in solving inverse problems. For instance, previous works demonstrate that self-supervised methods can capture intrinsic, high-level data structures without explicit labels, such as in partial differential equations [4] and computer vision tasks [3]. This latent structure can be effectively leveraged to regularize and improve the reconstruction process, even when direct supervision is limited.

Building on these ideas, our project proposes an organ-agnostic MRI reconstruction framework that integrates learned representations with physics-aware inversion. We begin by training a ResNet-based classifier [2] on labeled MRI images (knee vs. brain) to encode them into latent representation vectors. Preliminary results indicate that these representations form distinct clusters corresponding to each organ and suggest that the network successfully captures organ-specific features. We then integrate these learned representations into an end-to-end VarNet reconstruction model [5]. This integration aims to combine the benefits of a data-driven, self-supervised representation with the rigorous constraints of a physics-based forward operator to yield high-fidelity reconstructions that are robust across different anatomies.

This research is significant because it bridges the gap between self-supervised representation learning and physics-based inversion in MRI. By leveraging a shared latent space that naturally separates organ-specific features, our approach promises to streamline reconstruction pipelines, reduce the need for multiple specialized models, and enhance generalization to new anatomical structures and scanning protocols.

---

## 2. Objectives

The primary objectives of this project are:

1. **Develop a Unified Reconstruction Model**: Construct an organ-agnostic MRI reconstruction network that leverages a shared latent representation to handle multiple anatomies (brain, knee, prostate, etc.) within a single framework.

2. **Learn Robust Representations**: Train an encoder model in self-supervised manner to encode MRI images into latent representation vectors, which ensures that these vectors naturally cluster according to organ type.

3. **Integrate Representations into Reconstruction**: Incorporate the learned latent representations into a reconstruction pipeline by conditioning the network on organ-specific information. This fusion is intended to improve reconstruction accuracy by embedding organ-specific context while preserving the measurement consistency enforced by the forward operator $A$.

4. **Establish Evaluation Metrics and Generalization**: Evaluate the reconstruction quality using quantitative metrics (e.g., PSNR [1], SSIM [6]) and assess the model’s ability to generalize to new organ types and scanning protocols.

---

## 3. Approach and Method

Our approach builds upon the setting of the E2E-VarNet architecture [5], which reconstructs MRI images from undersampled multi-coil $k$-space data using a cascade of refinement modules. In this framework, the reconstruction is formulated as an iterative process where each cascade refines the current $k$-space estimate. We adopt this setting and extend it by incorporating organ-specific latent representations.

### Setting and Baseline Framework

We assume the standard MRI acquisition model [5]:

$$
  k = A(x) + \epsilon,
$$

where $k$ denotes the acquired $k$-space data, $A$ is the forward operator (including coil sensitivity maps, Fourier transform, and undersampling mask $M$), and $\epsilon$ represents noise. The baseline reconstruction model follows the E2E-VarNet formulation, where the update at cascade $t$ is given by:

$$
  k_{t+1} = k_{t} - \eta_{t} M(k_{t} - \tilde{k}) + G(k_{t}).
$$

Here, $G(\cdot)$ is the refinement module implemented via a U-Net (or similar CNN), and $\eta_{t}$ is a learned step-size parameter.

### Method Details

Our method extends the baseline framework by integrating a representation learning branch:

1. **Representation Learning**  
   - We train an encoder model on labeled MRI images of different organs to obtain latent representation vectors $z$.  
   - Preliminary experiments demonstrate that these vectors naturally cluster by organ type, serving as an informative prior for the reconstruction.

2. **Conditional Integration into Encoder-decoder Reconstruction Model**  
   - We modify the refinement module $G(\cdot)$ in each cascade to condition its operation on the latent representation $z$.  
   - Specifically, in the case of the E2E-VarNet, we propose to incorporate the latent vector via a conditioning mechanism within the CNN architecture. The modified refinement module can be expressed as:

$$
  G'(k_{t}, z) = F \circ \text{CNN}\Big(R \circ F^{-1}(k_{t}) \oplus \phi(z) \Big),
$$

  where $F$ and $F^{-1}$ denote the Fourier and inverse Fourier transforms, $R$ is the coil reduction operator, $\phi(z)$ is a learned mapping of the latent vector to a compatible feature space, and $\oplus$ denotes the conditioning operation.

3. **Evaluation**  
   - The integrated model will be evaluated on the fastMRI dataset using quantitative metrics (PSNR [1], SSIM [6]) and qualitative visual assessments.  
   - Ablation studies will be conducted to isolate the impact of the latent representation conditioning on reconstruction performance.

### Baseline Method

As a first milestone towards a unified reconstruction model, we propose the following construction as a baseline method:

1. We first establish the ability for a model to generalize to two organs—knee and brain—as a first step to progress towards multi-organ.

2. In contrast with the final objective of employing a self-supervised approach, we focus on proving effectiveness of latent representation by first using a supervised learning approach. A ResNet-18 model [2] is trained on a classification task (knee or brain), where the learned representation from training can be outputted as vectors by taking off the classification head. Our current result shows that the representation vectors mapped to the 2D plane via t-SNE demonstrate a clear separation of clusters between the knee and the brain, a preliminary evidence to their effectiveness in helping reconstruction models.

3. We stick to the E2E-VarNet for the reconstruction model. To condition the reconstruction network on organ-specific information, we extend the baseline model by integrating latent representation vectors directly into the regularizer. We will use direct concatenation to fuse the latent features with the network’s input. This vector is then projected via a linear layer to match the U-Net input channel dimension and is spatially expanded. The key operation is the direct concatenation of the expanded latent features with the normalized input tensor, as shown in the `NormUnet` module of our implementation, extending from the fastMRI implementation [7].

For example, the following code snippet (Listing 1) illustrates the core idea:

```python
# Process latent vector: project and reshape to match spatial dimensions

latent_features = self.latent_proj(latent_vector)  # [batch, in_chans]
latent_features = latent_features.view(*latent_features.shape, 1, 1)  # [batch, in_chans, 1, 1]
latent_features = latent_features.expand(-1, -1, x.shape[-2], x.shape[-1])  # [batch, in_chans, H, W]

# Concatenate latent features with input
x = torch.cat([x, latent_features], dim=1)  # [batch, in_chans*2, H, W]
```

This direct concatenation method is a straightforward demonstration of the general idea, and should allow the network to leverage additional organ-specific context without requiring complex modulation schemes.

---

## 4. Results

Below we present the set of results.


|  **Reconstruction vs. Ground Truth (Brain)**                                 |     **t-SNE of Latent Vectors (Brain vs. Knee)**                             |
| :--------------------------------------------------------------------------- | :-------------------------------------------------------------------- |
| ![vult-visualization_varnet-brain_leaderboard_state_dict](https://github.com/user-attachments/assets/e43c81a4-589f-4367-b9d0-0f751d86f11b)        | <img src="https://github.com/user-attachments/assets/01cc4be3-ec3b-499d-a1c1-7bd49c93d4f5" alt="Alt Text" width="700" > |

| **Knee vs. Brain SSIM Comparison**                                           |  **Knee vs. Brain PSNR Comparison**                          |
| :--------------------------------------------------------------------------- | :--------------------------------------------------------------------- |
| ![knee_vs_brain_ssim_comparison_benchmark_ifft_crop_fft](https://github.com/user-attachments/assets/8e3c5560-f3c8-4811-8594-b18aaec6ef1f)       | ![knee_vs_brain_psnr_comparison_benchmark_ifft_crop_fft](https://github.com/user-attachments/assets/9a31ed72-e2b2-486a-8c46-d416da494bf5)   |

---

### Explanations

1. **Reconstruction vs. Ground Truth**  
   - **Plot**: Side-by-side comparison of a ground-truth brain MRI slice (left) and its reconstructed counterpart (right).  
   - **Observations**: The major structural features match well, demonstrating that the latent-vector-conditioned model effectively captures the essential anatomy and yields a visually faithful result despite some minor background artifacts.

2. **t-SNE of Latent Vectors**  
   - **Plot**: Shows the 2D projection of latent vectors extracted from the encoder for knee vs. brain MRI slices.  
   - **Interpretation**: The distinct purple and yellow clusters illustrate that the network encodes knee vs. brain data into clearly separable latent representations, indicating a robust organ-specific embedding.


3. **Knee vs. Brain SSIM Comparison**  
   - **Plot**: SSIM (Structural Similarity Index Measure) for knee (x-axis) vs. brain (y-axis).  
   - **Finding**: Again, **latent_vanilla** shows improved performance on both organs, while single-organ models are biased toward their respective training organ (brain or knee).
  
4. **Knee vs. Brain PSNR Comparison**  
   - **Plot**: PSNR (Peak Signal-to-Noise Ratio) of knee images (x-axis) vs. brain images (y-axis) for three different models:
     - **knee_pretrained**: Model trained mainly on knee data  
     - **brain_pretrained**: Model trained mainly on brain data  
     - **latent_vanilla**: Organ-agnostic model that injects latent vectors  
   - **Finding**: Injection of latent vectors ("latent_vanilla") helps the model achieve balanced, higher PSNR across both brain and knee images, outperforming single-organ pretrained baselines.


