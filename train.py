import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
import pandas as pd
import json
import requests
import time
import os
import pickle
from tqdm import tqdm
from model import GenePanelGNN
from sklearn.metrics import roc_auc_score, accuracy_score


def load_gene_data_from_api():
    """Load gene-panel mappings from PanelApp API"""
    print("Fetching gene data from PanelApp API...")
    
    # Fetch all genes from PanelApp
    genes, url = [], "https://panelapp.genomicsengland.co.uk/api/v1/genes/?page_size=1000"
    while url:
        response = requests.get(url, timeout=30)
        data = response.json()
        genes += [g["gene_data"]["gene_symbol"] for g in data["results"] if g.get("entity_type") == "gene"]
        url = data["next"]
    
    print(f"Found {len(set(genes))} unique genes")
    
    # Create gene-to-panel mappings
    gene2panel = {}
    unique_genes = list(set(genes))
    
    for gene in tqdm(unique_genes, desc="Processing genes"):
        if gene in gene2panel:
            continue
        
        time.sleep(0.1)  # Rate limiting
        try:
            response = requests.get(f"https://panelapp.genomicsengland.co.uk/api/v1/entities/{gene}/")
            panels = [x['panel']['id'] for x in response.json()['results']]
            gene2panel[gene] = panels
        except Exception as e:
            print(f"Error processing gene {gene}: {e}")
            gene2panel[gene] = []
    
    return gene2panel


def load_gene_data_from_cache():
    """Load gene data from cache if available, otherwise fetch from API"""
    cache_file = "gene2panel_cache.pkl"
    
    if os.path.exists(cache_file):
        print("Loading gene data from cache...")
        with open(cache_file, 'rb') as f:
            gene2panel = pickle.load(f)
    else:
        gene2panel = load_gene_data_from_api()
        print("Saving gene data to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(gene2panel, f)
    
    return gene2panel


def create_graph_data(gene2panel):
    """Create PyTorch Geometric HeteroData from gene-panel mappings"""
    print("Creating graph data structure...")
    
    # Create edges DataFrame
    edges_df = pd.DataFrame(
        [(gene, panel) for gene, panels in gene2panel.items() for panel in panels], 
        columns=["gene", "panel"]
    )
    
    # Create index mappings
    gene2idx = {gene: i for i, gene in enumerate(sorted(edges_df["gene"].unique()))}
    panel2idx = {panel: i for i, panel in enumerate(sorted(edges_df["panel"].unique()))}
    idx2panel = {i: panel for panel, i in panel2idx.items()}
    
    # Create edge indices
    edges_df["gene_idx"] = edges_df["gene"].map(gene2idx)
    edges_df["panel_idx"] = edges_df["panel"].map(panel2idx)
    edge_index = torch.tensor(edges_df[["gene_idx", "panel_idx"]].values.T, dtype=torch.long)
    
    # Create HeteroData object
    data = HeteroData()
    data['gene'].num_nodes = len(gene2idx)
    data['panel'].num_nodes = len(panel2idx)
    
    # Add edges (both directions for heterogeneous graph)
    data['gene', 'in', 'panel'].edge_index = edge_index
    data['panel', 'rev_in', 'gene'].edge_index = edge_index.flip(0)
    
    # Initialize node features as learnable embeddings
    embedding_dim = 128
    data['gene'].x = torch.nn.Embedding(len(gene2idx), embedding_dim).weight
    data['panel'].x = torch.nn.Embedding(len(panel2idx), embedding_dim).weight
    
    print(f"Created graph with {len(gene2idx)} genes, {len(panel2idx)} panels, {edge_index.size(1)} edges")
    
    return data, gene2idx, panel2idx, idx2panel


def train_model(data, gene2idx, panel2idx, epochs=100, lr=1e-3, hidden_dim=128):
    """Train the GNN model"""
    print("Starting model training...")
    
    # Initialize model
    model = GenePanelGNN(hidden=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Prepare data dictionaries
    x_dict = {
        'gene': data['gene'].x,
        'panel': data['panel'].x
    }
    edge_index_dict = {
        ('gene', 'in', 'panel'): data['gene', 'in', 'panel'].edge_index,
        ('panel', 'rev_in', 'gene'): data['panel', 'rev_in', 'gene'].edge_index
    }
    
    pos_edge_index = data['gene', 'in', 'panel'].edge_index
    n_genes, n_panels = data['gene'].num_nodes, data['panel'].num_nodes
    
    model.train()
    for epoch in range(epochs):
        # Sample negative edges
        neg_edges = negative_sampling(
            pos_edge_index, 
            num_nodes=(n_genes, n_panels),
            num_neg_samples=pos_edge_index.size(1)
        )
        
        # Forward pass
        pos_score = model(x_dict, edge_index_dict, pos_edge_index)
        neg_score = model(x_dict, edge_index_dict, neg_edges)
        
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]),
            torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model


def evaluate_model(model, data):
    """Evaluate the trained model"""
    print("Evaluating model...")
    
    # Split data for evaluation
    split = RandomLinkSplit(
        num_val=0.10,
        num_test=0.10,
        edge_types=[('gene', 'in', 'panel')],
        rev_edge_types=[('panel', 'rev_in', 'gene')],
        add_negative_train_samples=False,
        is_undirected=False
    )
    
    train_data, val_data, test_data = split(data)
    
    # Prepare data for evaluation
    x_dict = {
        'gene': data['gene'].x,
        'panel': data['panel'].x
    }
    edge_index_dict = {
        ('gene', 'in', 'panel'): train_data['gene', 'in', 'panel'].edge_index,
        ('panel', 'rev_in', 'gene'): train_data['panel', 'rev_in', 'gene'].edge_index
    }
    
    model.eval()
    with torch.no_grad():
        # Test on validation edges
        val_pos_edges = val_data['gene', 'in', 'panel'].edge_index
        val_neg_edges = negative_sampling(
            val_pos_edges,
            num_nodes=(data['gene'].num_nodes, data['panel'].num_nodes),
            num_neg_samples=val_pos_edges.size(1)
        )
        
        # Get predictions
        pos_scores = torch.sigmoid(model(x_dict, edge_index_dict, val_pos_edges))
        neg_scores = torch.sigmoid(model(x_dict, edge_index_dict, val_neg_edges))
        
        # Calculate metrics
        y_true = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu().numpy()
        y_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        y_pred = (y_scores > 0.5).astype(int)
        
        auc = roc_auc_score(y_true, y_scores)
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"Validation AUC: {auc:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
    
    return auc, accuracy


def save_checkpoint(model, data, gene2idx, panel2idx, idx2panel, filepath='gnn_panelapp.pt'):
    """Save model checkpoint with all necessary data"""
    print(f"Saving checkpoint to {filepath}...")
    
    checkpoint = {
        'model': model.state_dict(),
        'data': data,
        'gene2idx': gene2idx,
        'panel2idx': panel2idx,
        'idx2panel': idx2panel
    }
    
    torch.save(checkpoint, filepath)
    print("Checkpoint saved successfully!")


def main():
    """Main training pipeline"""
    print("Starting GNN training pipeline...")
    
    # Load data
    gene2panel = load_gene_data_from_cache()
    
    # Create graph
    data, gene2idx, panel2idx, idx2panel = create_graph_data(gene2panel)
    
    # Train model
    model = train_model(data, gene2idx, panel2idx, epochs=100)
    
    # Evaluate model
    auc, accuracy = evaluate_model(model, data)
    
    # Save checkpoint
    save_checkpoint(model, data, gene2idx, panel2idx, idx2panel)
    
    print("Training completed successfully!")
    print(f"Final metrics - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()