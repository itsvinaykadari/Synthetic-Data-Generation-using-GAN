# Synthetic Data Generation using WGAN-GP

## Problem Statement

**Challenge**: Real-world fraud detection datasets are often limited, imbalanced, and sensitive to privacy concerns. Training machine learning models on limited data leads to poor generalization, overfitting, and inability to capture diverse fraud patterns.

**Solution**: Generate high-quality synthetic fraud detection data that:
1. **Preserves Statistical Properties**: Maintains feature distributions and statistical characteristics
2. **Maintains Relationships**: Retains correlations and dependencies between features
3. **Ensures Authenticity**: Produces data indistinguishable from real samples
4. **Protects Privacy**: Enables data augmentation without exposing sensitive information

**Why WGAN-GP?**
- **Stable Training**: Wasserstein loss provides meaningful gradients throughout training
- **No Mode Collapse**: Gradient penalty prevents generator from producing limited variety
- **Better Convergence**: Lipschitz constraint ensures smooth learning surface
- **Feature Preservation**: Effectively captures continuous feature distributions

## Project Highlights

- Custom implementation of **Generator** and **Discriminator** networks from scratch
- **Wasserstein loss with gradient penalty** for stable adversarial training
- Comprehensive evaluation using **Jensen-Shannon Divergence**, KDE plots, and correlation analysis
- Practical utility testing through **classification-based discriminator test**

## Dataset

- **Input file**: `data.xlsx`
- **Instances**: 1,199 samples
- **Features**: 10 continuous numerical attributes

### 📊 Download Dataset

Access the complete dataset on Google Drive:
👉 **[Dataset Link](https://docs.google.com/spreadsheets/d/1GF8kOx7cxfTYXCicC3IBQKTLnduKsUGc/edit?usp=sharing&ouid=116235528412066039197&rtpof=true&sd=true)**

**To use the dataset**:
1. Click the link above to open in Google Drive
2. Download as `.xlsx` or `.csv` file
3. Place `data.xlsx` in the project root directory
4. Run the notebook

## Setup Instructions

1. **Install Dependencies**

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn scipy
```

2. **Run Notebook**

Open and run `WGAN_GP_Synthetic_Data_Generation.ipynb` to:

- Preprocess and normalize real data
- Train WGAN-GP model (1200 epochs)
- Generate 1,199 synthetic samples
- Evaluate quality through multiple metrics

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

### 1. **Distribution Analysis**
- **KDE Plots**: Visual comparison of feature distributions (real vs synthetic)
- **Jensen-Shannon Divergence**: Quantitative measure of distribution similarity per feature
  - Range: 0 (identical) to 1 (completely different)
  - Lower JSD indicates better synthetic data quality

### 2. **Correlation Preservation**
- **Pearson Correlation Matrices**: Heatmaps showing feature relationships
- **Correlation Similarity Score**: Measures preservation of feature dependencies
  - Score near 1.0 = excellent preservation
  - Critical for applications requiring correlated features

### 3. **Realism Assessment**
- **Classifier Discriminator Test**: Trains LogisticRegression to distinguish real vs synthetic
  - **~50% accuracy** = synthetic data is indistinguishable (excellent)
  - **>65% accuracy** = synthetic data is distinguishable (poor quality)
  - Tests practical utility of generated samples

### 4. **Training Dynamics**
- **Loss Curves**: Monitors Generator and Critic convergence
- Stable convergence indicates successful WGAN-GP training

## Model Architecture

### Generator
- Input: 32-dimensional random noise vector
- Hidden layers: 128 → 256 neurons with ReLU activation
- Output: Data-dimensional samples with Tanh activation [-1, 1]
- Purpose: Transforms noise into realistic synthetic samples

### Discriminator (Critic)
- Input: Real or synthetic data samples
- Hidden layers: 256 → 128 neurons with LeakyReLU(0.2) activation
- Output: Single neuron (Wasserstein distance estimate)
- Purpose: Distinguishes real from fake and guides generator training

### Gradient Penalty
- Enforces 1-Lipschitz constraint on discriminator
- Interpolates between real and fake samples
- Weight parameter: λ = 10 (balances penalty strength)

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| latent_dim | 32 | Size of noise input to generator |
| batch_size | 64 | Provides stable gradient estimates |
| n_critic | 10 | Discriminator updates per generator update |
| lambda_gp | 10 | Gradient penalty weight |
| epochs | 1200 | Training iterations for convergence |
| lr_generator | 8e-5 | Learning rate for generator |
| lr_discriminator | 5e-5 | Learning rate for discriminator |

## Key Features

✅ **Stable Training**: WGAN-GP prevents mode collapse and divergence  
✅ **Distribution Preservation**: Maintains statistical properties of real data  
✅ **Correlation Retention**: Preserves feature relationships  
✅ **Privacy-Friendly**: Generates new samples, not memorized data  
✅ **Scalable**: Handles continuous features typical in fraud detection  

## Future Improvements

- Progressive training for handling larger datasets
- Conditional generation (fraud vs non-fraud)
- Integration with anomaly detection pipelines
- Hyperparameter optimization through grid search
- Model persistence and checkpointing
- Real-time monitoring with TensorBoard
- Additional evaluation metrics (Inception Score, FID)
