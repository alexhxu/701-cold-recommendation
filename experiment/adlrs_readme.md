# ADLRS Model

## Overview

This implementation constructs Attention-based Deep Learning Recommender System (ADLRS) based on the paper "An attention-based deep learning method for solving the cold-start and sparsity issues of recommender systems".

## Code Reference

### Core Functions

#### `run_adlrs_pipeline()`
Main training pipeline that runs all steps.

**Parameters:**
- `user_ids` (array): User IDs for ratings
- `item_ids` (array): Item IDs for ratings
- `ratings` (array): Rating values
- `item_profiles_dict` (dict): Item ID to profile text mapping
- `dae_file` (str, optional): Path to pre-trained DAE model
- `num_factors` (int): Latent factor dimension (default: 20)
- `k_neighbors` (int): Number of similar items (default: 3)
- `z_gamma` (float): Gamma parameter (default: 5.0)
- `mf_epochs` (int): Training epochs for MF (default: 20)
- `device` (str): 'cpu' or 'cuda'

**Returns:** Dict with trained model, embeddings, and mappings

#### `save_adlrs_model()`
Save complete ADLRS model.

**Parameters:**
- `result` (dict): Output from `run_adlrs_pipeline()`
- `filepath` (str): Save path
- `config` (dict, optional): Configuration metadata

#### `load_adlrs_model()`
Load saved ADLRS model.

**Parameters:**
- `filepath` (str): Path to saved model
- `device` (str): 'cpu' or 'cuda'

**Returns:** Dict with model and metadata

#### `predict_ratings()`
Predict ratings for user-item pairs.

**Parameters:**
- `model`: Trained MF model
- `user_ids` (array): User IDs to predict
- `item_ids` (array): Item IDs to predict
- `user2idx` (dict): User ID mapping
- `item2idx` (dict): Item ID mapping
- `device` (str): 'cpu' or 'cuda'

**Returns:** numpy array of predictions

### Model Classes

#### `DeepAutoEncoder`
Deep autoencoder for dimensionality reduction.

#### `MF`
Matrix Factorization with ADLRS regularization.

### Evaluation Metrics

- Mean Absolute Error
- Root Mean Squared Error
- Hit Ratio @ N

## Training Workflow

### Standard Workflow
1. **Data Preparation**: Load ratings and item profiles
2. **BERT Embedding**: Convert item profiles to 768-d vectors
3. **DAE Training**: Reduce to lower dimension (e.g., 20-d)
4. **MF Training**: Train matrix factorization with side info
5. **Evaluation**: Test on held-out data

### Efficient Workflow (Reusing DAE)
1. **Train DAE once** and save it
2. **Experiment with MF** using different hyperparameters
3. **Reuse same item embeddings** without retraining DAE

## Hyperparameters

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_factors` | 20 | Latent factor dimension |
| `k_neighbors` | 3 | Number of similar items for regularization |
| `z_gamma` | 5.0 | Controls cold-start vs rating-based learning |
| `beta` | 0.01 | L2 regularization strength |
| `lr` | 0.01 | Learning rate for MF |
| `dae_epochs` | 10 | Training epochs for DAE |
| `mf_epochs` | 20 | Training epochs for MF |

## File Structure

```
ADLRS.ipynb             # all implementation and running pipeline
trained_dae.pt          # Saved DAE model
adlrs_complete_model.pt # Saved complete model
```

## Citation
```
Heidari, N., Moradi, P., & Koochari, A. (2022). 
An attention-based deep learning method for solving the cold-start 
and sparsity issues of recommender systems. 
Knowledge-Based Systems, 256, 109835.
```
