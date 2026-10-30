# Text-to-Image Generation Using Stable Diffusion

### Practical 5 | Generative AI | Hugging Face Diffusers

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Diffusers-yellow)](https://huggingface.co/docs/diffusers)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-orange)](https://colab.research.google.com/drive/1PE0QNNn4IC12kOVUeGYvZ1zwc0phZs_8?usp=sharing)

## 1. Project Overview

This project implements a pretrained **Stable Diffusion v1.5** model using the Hugging Face Diffusers library to generate images from natural language text prompts.

The main objective is to understand how different inference parameters influence the quality, visual appearance, prompt alignment, convergence, and generation time of AI-generated images.

The project systematically investigates three important factors:

- Number of inference steps.
- Classifier-Free Guidance (CFG) scale.
- Different sampling methods (Schedulers).

It also studies the interaction between inference steps and guidance scale to understand how these parameters work together.

The generated images are evaluated using quantitative metrics, visual comparisons, graphs, and performance measurements.

## 2. Project Objectives

1. Implement a pretrained Stable Diffusion model using Hugging Face Diffusers.
2. Generate images from different natural language text prompts.
3. Analyze the effect of different inference step counts on image generation.
4. Study how guidance scale affects prompt adherence and visual characteristics.
5. Compare different sampling algorithms and their performance.
6. Evaluate generated images using CLIP similarity, PSNR, sharpness, contrast, and saturation.
7. Analyze the relationship between image quality and generation time.
8. Visualize experimental results using graphs, comparison grids, and heatmaps.

## 3. Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Hugging Face Diffusers | Stable Diffusion implementation |
| Transformers | CLIP model and processor |
| Stable Diffusion v1.5 | Pretrained text-to-image model |
| NumPy | Numerical computations |
| Pandas | Experimental data analysis |
| Matplotlib | Graphs and visualizations |
| PIL (Pillow) | Image processing |
| SciPy | Image sharpness calculations |
| Google Colab | Cloud-based execution with GPU |

## 4. Model Architecture

Stable Diffusion is a latent diffusion model that generates images by gradually removing noise from a latent representation.

Unlike traditional pixel-based diffusion approaches, Stable Diffusion performs the diffusion process in a compressed latent space.

### Main Components

**1. CLIP Text Encoder**

- Converts the input text prompt into numerical embeddings.
- Captures the semantic meaning of the prompt.
- Provides text conditioning to the U-Net.

**2. U-Net**

- Predicts the noise present in the latent representation.
- Iteratively removes noise during the inference process.
- Uses text embeddings to guide image generation.

**3. VAE Decoder**

- Converts the final denoised latent representation into a visible image.
- Produces the final RGB image.

**4. Scheduler**

- Controls the reverse diffusion process.
- Determines how the noise is removed at each inference step.
- Different schedulers use different numerical methods to update the latent representation.

### Generation Workflow

```text
          Text Prompt
               |
               v
        CLIP Text Encoder
               |
               v
        Text Embeddings
               |
               v
       Random Noise Latent
               |
               v
       +---------------+
       |     U-Net     |
       | Noise Prediction|
       +---------------+
               |
               v
           Scheduler
               |
       Iterative Denoising
               |
               v
       Final Latent Image
               |
               v
          VAE Decoder
               |
               v
        Generated Image
               |
               v
       Image Evaluation
```

## 5. Pretrained Model

The project uses the Stable Diffusion v1.5 checkpoint available on Hugging Face.

**Model:** `stable-diffusion-v1-5/stable-diffusion-v1-5`

Model link:

https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

The pretrained model is loaded using the `StableDiffusionPipeline` class from the Diffusers library.

The implementation uses:

- 512 × 512 image resolution.
- Fixed random seed: 42.
- Float16 precision when running on CUDA.
- Float32 precision when running on CPU.
- Negative prompting to reduce unwanted visual characteristics.

## 6. Experimental Setup

Three different prompts are used to evaluate image generation across different visual styles.

| Prompt | Description | Style |
|---|---|---|
| Astronaut | An astronaut riding a horse on Mars | Photograph |
| Cabin | A cozy wooden cabin in a snowy forest at sunset | Oil painting |
| Ramen | A bowl of ramen with a soft-boiled egg | Food photography |

**Negative Prompt:**

`blurry, low quality, distorted, extra limbs`

A fixed seed is used to improve reproducibility and make comparisons more meaningful.

## 7. Experiments Performed

### Experiment 1: Effect of Inference Steps

**Objective:** Analyze how the number of inference steps affects image quality and generation time.

The number of inference steps controls how many denoising iterations are performed during image generation.

**Experimental Settings:**

| Parameter | Value |
|---|---|
| Inference steps | 5, 10, 15, 20, 30, 50, 100 |
| Guidance scale | 7.5 |
| Sampling method | DPM++ 2M |
| Prompts | 3 |

The 100-step output is used as the reference image for convergence comparisons.

**Evaluation:**
- Visual quality and detail.
- CLIP similarity score.
- PSNR against the reference image.
- Image sharpness.
- Generation time.

**Expected Analysis:**

A lower number of steps generally produces images faster but may leave details insufficiently refined. Increasing the number of steps can improve convergence, although additional steps eventually provide diminishing returns.

### Experiment 2: Effect of Guidance Scale

**Objective:** Study how Classifier-Free Guidance affects the relationship between the generated image and the input text prompt.

Guidance scale controls how strongly the model follows the text prompt.

**Experimental Settings:**

| Parameter | Value |
|---|---|
| Inference steps | 30 |
| Guidance scale | 1, 3, 5, 7.5, 10, 15, 20, 30 |
| Sampling method | DPM++ 2M |
| Prompts | 3 |

**Evaluation:**
- CLIP similarity score.
- Image saturation.
- Contrast.
- Sharpness.
- Visual appearance.

**Expected Analysis:**

A low guidance scale may produce images that do not closely follow the prompt.

A moderate guidance scale can provide a useful balance between prompt adherence and natural appearance.

Very high guidance values can cause excessive saturation, harsh contrast, and visual artefacts.

### Experiment 3: Comparison of Sampling Methods

**Objective:** Compare different schedulers to understand their impact on image quality, convergence, and generation speed.

A scheduler determines how the latent representation is updated during the reverse diffusion process.

**Nine Sampling Methods Compared:**

| Sampling Method | Description |
|---|---|
| PNDM | Pseudo-numerical diffusion method |
| DDIM | Denoising Diffusion Implicit Models |
| LMS | Linear Multistep Scheduler |
| Euler | First-order numerical solver |
| Euler A | Ancestral Euler sampler |
| Heun | Second-order numerical solver |
| DPM++ 2M | Second-order multistep solver |
| DPM++ 2M Karras | DPM++ 2M with Karras noise spacing |
| UniPC | Predictor-corrector scheduler |

**Experimental Settings:**

- Multiple inference step counts.
- Fixed guidance scale.
- Same prompts and random seed.
- Comparison against a 100-step DPM++ 2M reference.

**Evaluation:**
- PSNR convergence.
- CLIP similarity.
- Generation time.
- Visual quality.
- Overall scheduler ranking.

The experiment also generates heatmaps and convergence curves to compare scheduler performance.

### Experiment 4: Interaction Between Inference Steps and Guidance Scale

**Objective:** Understand how inference steps and guidance scale jointly affect the generated image.

Instead of changing one parameter at a time, this experiment evaluates combinations of both parameters.

**Experimental Settings:**

Inference steps:

`[8, 15, 30, 50]`

Guidance scale:

`[3, 7.5, 12, 20]`

Sampling method: DPM++ 2M.

Prompt: Cozy wooden cabin in a snowy forest.

A 4 × 4 comparison grid is generated.

**Evaluation:**
- CLIP similarity.
- Sharpness.
- Saturation.
- Visual quality.

This experiment helps identify combinations that provide a suitable balance between prompt adherence and image quality.

## 8. Evaluation Metrics

Several quantitative metrics are used to evaluate the generated images.

### 1. CLIP Similarity Score

CLIP compares image embeddings with text embeddings to estimate how closely the generated image matches the input prompt.

The image and text embeddings are normalized before calculating cosine similarity.

Higher similarity generally indicates better semantic alignment.

### 2. Peak Signal-to-Noise Ratio (PSNR)

PSNR measures the similarity between a generated image and a reference image.

In this project, the 100-step DPM++ 2M image is used as the reference for the same prompt and seed.

Higher PSNR indicates greater similarity to the reference.

**Note:** PSNR measures convergence to the chosen reference, not absolute image quality.

### 3. Sharpness

Sharpness is estimated using the variance of the Laplacian of the grayscale image.

Higher values generally indicate more high-frequency image detail, but excessive sharpness can also represent noise or artefacts.

### 4. Contrast

Contrast is calculated using the standard deviation of grayscale pixel intensity.

It helps analyze changes in the distribution of brightness values.

### 5. Saturation

Saturation is calculated from the HSV representation of the generated image.

It measures the intensity of colours in the image.

### 6. Generation Time

Generation time measures the time required to generate an image.

It is recorded in seconds and used to compare the computational efficiency of different inference settings and schedulers.

## 9. Results and Visualizations

The notebook automatically generates experimental comparison grids, quantitative tables, graphs, and heatmaps.

The main visualizations include:

- Inference steps comparison grids.
- Zoomed image details.
- Difference maps against the reference.
- Guidance scale comparison grids.
- Sampler comparison grids.
- Scheduler convergence curves.
- Quality versus generation time plots.
- Sampler performance heatmaps.
- Steps and guidance interaction heatmaps.
- Consolidated experimental overview.

All numerical observations and conclusions are generated from the actual experimental measurements.

**Important:** Actual performance values depend on the execution environment and should be taken from the generated notebook outputs rather than assumed in advance.

## 10. Project Structure

```text
Stable-Diffusion-Practical-5/
│
├── README.md
│
├── Practical_5_Stable_Diffusion_Fixed.ipynb
│
├── sd_results/
│   ├── exp1_steps_p0.png
│   ├── exp1_steps_p1.png
│   ├── exp1_steps_p2.png
│   ├── exp1_zoom.png
│   ├── exp1_diffmaps.png
│   ├── exp1_summary.png
│   │
│   ├── exp2_guidance_p0.png
│   ├── exp2_guidance_p1.png
│   ├── exp2_guidance_p2.png
│   ├── exp2_zoom.png
│   │
│   ├── exp3_heatmaps.png
│   ├── exp3_convergence_efficiency.png
│   │
│   ├── exp4_interaction_grid.png
│   ├── exp4_heatmaps.png
│   │
│   ├── overview.png
│   ├── results_all.csv
│   └── sampler_ranking.csv
│
└── requirements.txt
```

*The `sd_results/` folder and its files are generated after successfully running the notebook.*

## 11. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sanika13112/Stable-Diffusion-Practical-5.git

cd Stable-Diffusion-Practical-5
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install the required libraries directly:

```bash
pip install diffusers transformers accelerate safetensors scipy
pip install torch torchvision
pip install numpy pandas matplotlib pillow
```

### Step 3: Open Google Colab

Open the notebook using the link below:

https://colab.research.google.com/drive/1PE0QNNn4IC12kOVUeGYvZ1zwc0phZs_8?usp=sharing

### Step 4: Configure GPU

In Google Colab:

1. Open Runtime.
2. Select Change runtime type.
3. Choose T4 GPU if available.
4. Save the settings.

### Step 5: Execute the Notebook

Run the notebook cells sequentially.

The notebook performs the following operations:

1. Installs the required libraries.
2. Imports the dependencies.
3. Loads the pretrained Stable Diffusion model.
4. Generates a basic sample image.
5. Loads CLIP for image-text similarity evaluation.
6. Executes all four experiments.
7. Generates comparison graphs and heatmaps.
8. Calculates evaluation metrics.
9. Produces the consolidated results table.
10. Generates a ZIP file containing experimental outputs.

**Execution time:** Approximately 25–35 minutes on the notebook's recommended T4 GPU setup. Actual execution time may vary depending on GPU availability, model download time, and runtime conditions.

## 12. Reproducibility

To make the experiments more consistent, the following settings are maintained:

- Fixed random seed: 42.
- Same three text prompts.
- Same negative prompt.
- Same 512 × 512 output resolution.
- Consistent model checkpoint.
- Controlled inference parameters.
- Common reference image for convergence analysis.

A fixed seed improves reproducibility, although exact results can still vary across hardware, library versions, and execution environments.

## 13. Key Observations

The experiments are designed to investigate the following relationships:

| Parameter | Main Observation |
|---|---|
| Inference steps | More steps can improve refinement but increase generation time |
| Guidance scale | Higher guidance can improve prompt adherence but may introduce visual artefacts |
| Sampling methods | Different schedulers have different convergence and speed characteristics |
| Steps × Guidance | Both parameters jointly influence the final image |

The final observations, metric values, and recommended settings are generated automatically after the notebook completes execution.

## 14. Limitations

1. The experiments use only three text prompts.
2. A single random seed is used for controlled comparisons.
3. CLIP similarity measures semantic alignment, not complete visual aesthetics.
4. PSNR measures similarity to the selected reference image rather than absolute quality.
5. Sharpness, contrast, and saturation are numerical proxies and cannot fully represent human visual perception.
6. Results are specific to Stable Diffusion v1.5 and 512 × 512 image generation.
7. Generation time depends on the available GPU and software environment.

## 15. Conclusion

This project demonstrates text-to-image generation using a pretrained Stable Diffusion model and the Hugging Face Diffusers library.

Through four experiments, it investigates the impact of inference steps, guidance scale, and sampling methods on generated image quality and computational performance.

The use of CLIP similarity, PSNR, sharpness, contrast, saturation, and generation time provides a quantitative approach to understanding image generation behaviour.

The experiments also demonstrate that image generation is a balance between prompt adherence, visual refinement, convergence, and computational efficiency.

Overall, the project provides practical experience with latent diffusion models, pretrained generative AI systems, scheduler selection, image evaluation, and experimental analysis.

## 16. References

1. Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
2. Ho and Salimans (2022). Classifier-Free Diffusion Guidance.
3. Song et al. (2021). Denoising Diffusion Implicit Models. ICLR 2021.
4. Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. NeurIPS 2022.
5. Lu et al. (2022). DPM-Solver++.
6. Zhao et al. (2023). UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models.
7. Radford et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.
8. Hugging Face Diffusers Documentation: https://huggingface.co/docs/diffusers
9. Stable Diffusion v1.5 Model: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

---

## 17. Project Links

**GitHub Repository:**  
https://github.com/Sanika13112/Stable-Diffusion-Practical-5

**Google Colab Notebook:**  
https://colab.research.google.com/drive/1PE0QNNn4IC12kOVUeGYvZ1zwc0phZs_8?usp=sharing

---

**Author:** Sanika Mane

**Project:** Practical 5 – Text-to-Image Generation Using Stable Diffusion

**Domain:** Artificial Intelligence and Machine Learning | Generative AI
