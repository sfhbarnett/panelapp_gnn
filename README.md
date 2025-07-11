# PanelApp GNN

A Graph Neural Network (GNN) for predicting gene-panel associations using PanelApp data. This project implements a heterogeneous GNN to suggest which genomic panels a gene should be included in, helping with clinical genomics panel curation.

## Overview

This project uses PyTorch Geometric to build a bipartite graph connecting genes to genomic panels from PanelApp (Genomics England's clinical interpretation resource). The GNN learns embeddings for both genes and panels to predict likely gene-panel associations.

## Features

- **Heterogeneous GNN**: Uses HeteroConv with SAGEConv layers for gene-panel link prediction
- **Interactive Web Interface**: Streamlit app for real-time predictions
- **Dynamic Graph Updates**: Support for adding new genes and edges during inference
- **PanelApp Integration**: Data sourced from PanelApp API with 6,342 genes and 431 panels
- **Automated Training**: Standalone training script with evaluation metrics
- **Data Caching**: Efficient caching system to avoid repeated API calls

## Files

- `main.py` - Streamlit web interface for predictions
- `model.py` - GNN model definition (GenePanelGNN class)
- `train.py` - Training script for the GNN model
- `panelappapi.ipynb` - Data collection and exploration notebook
- `gnn_panelapp.pt` - Trained model weights and graph data
- `panelid2name.json` - Mapping of panel IDs to descriptive names
- `requirements.txt` - Python dependencies

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train a new model from scratch:

```bash
python train.py
```

This will:
- Fetch gene-panel data from PanelApp API (cached for subsequent runs)
- Create the heterogeneous graph structure
- Train the GNN model for 100 epochs
- Evaluate performance with AUC and accuracy metrics
- Save the trained model checkpoint

### Running the Web Interface

To use the interactive prediction interface:

```bash
streamlit run main.py
```

The interface allows you to:
1. Select one or more genomic panels from the dropdown
2. Click "Predict" to see the top 10 recommended additional panels
3. Results show panel names and confidence scores

## Model Architecture

- **Input**: Bipartite graph with gene and panel nodes
- **Layers**: 2 HeteroConv layers with SAGEConv message passing
- **Hidden Dimension**: 128-dimensional embeddings
- **Output**: Link prediction scores via bilinear scoring function
- **Training**: Binary cross-entropy loss with negative sampling

## Data

- **Genes**: 6,342 unique gene symbols from PanelApp
- **Panels**: 431 genomic panels covering various clinical conditions
- **Edges**: 35,280 gene-panel associations
- **Features**: Learned embeddings (128-dimensional)

## Training Configuration

The training script supports customizable parameters:
- **Epochs**: 100 (default)
- **Learning Rate**: 1e-3
- **Hidden Dimension**: 128
- **Evaluation Split**: 10% validation, 10% test

## Example Output

The model suggests panels like:
- "484: DDG2P" (score: 0.146)
- "162: Severe microcephaly" (score: 0.015)
- "402: Early onset or syndromic epilepsy" (score: 0.004)

## Performance

Model evaluation includes:
- **AUC Score**: Area under the ROC curve for link prediction
- **Accuracy**: Binary classification accuracy at 0.5 threshold
- **Validation Split**: Independent test set for unbiased evaluation