# Synthetic Data Generation using GAN

This project demonstrates the generation of synthetic data using a **Wasserstein GAN with Gradient Penalty (WGAN-GP)**, trained from scratch using PyTorch without predefined model libraries. The goal is to create synthetic data that closely mimics a real dataset while preserving feature distributions and correlations.

## Project Highlights

- Custom implementation of **Generator** and **Discriminator** networks.
- Use of **Wasserstein loss with gradient penalty** for stable training.
- Evaluation using **Jensen-Shannon Divergence**, KDE plots, and **Pearson correlation**.
- Additional classification and **outlier detection** to validate realism of synthetic data.

## Dataset

- **Input file**: `data.xlsx`
- **Instances**: 1,199
- **Features**: 10 continuous numerical attributes
- - Features include: `cov1`–`cov7`, `sal_pur_rat`, `igst_itc_tot_itc_rat`, `lib_igst_itc_rat`

## Setup Instructions

1. **Install Dependencies**

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn
```

2. **Run Notebook**

Open and run `Fraud_Assignment_4.ipynb` to:

- Preprocess data
- Train WGAN-GP
- Generate synthetic data
- Visualize evaluation metrics

## Model Architecture

### Generator
- Input: Latent vector (random noise)
- Fully Connected layers
- Activation: ReLU (hidden), Tanh (output)

### Discriminator (Critic)
- Input: Real or synthetic data
- Fully Connected layers
- Activation: LeakyReLU
- Output: Authenticity score

## Evaluation Metrics

- **Jensen-Shannon Divergence**: For feature-wise distribution similarity
- **KDE Plots**: Compare real vs synthetic feature distributions
- **Pearson Correlation Heatmaps**: Compare inter-feature relationships
- **Classifier Accuracy**: Discriminator trained to distinguish real/synthetic (≈50.28%)
- **Outlier Detection**: Detect both global (sparse) and boundary anomalies in generated data

## Results

- Real and synthetic distributions are statistically similar.
- Correlation Similarity Score: **95.62**
- Classifier performance close to random (50.28%), confirming data realism.
- Visualizations support both global and local outlier analysis.

## Future Work

- Adapt to domain-specific data (e.g., fraud, healthcare)
- Optimize training efficiency
- Enhance evaluation with more robust anomaly detection

## Contributors

- Vinaykumar Kadari (CS24MTECH14008)
- Chaudhary Khushbu Rakesh (CS24MTECH14012)
- Challa Sri Tejaswini (CS24MTECH14016)
