import streamlit as st
import torch
from model import GenePanelGNN
import pandas as pd
import json



st.title("GNN Predictor")

# Load model once
ckpt = torch.load('gnn_panelapp.pt', map_location='cpu', weights_only=False)
model = GenePanelGNN()  # supply init args
model.load_state_dict(ckpt['model'])
gene2idx  = ckpt['gene2idx'];  panel2idx = ckpt['panel2idx']
data = ckpt['data']
idx2panel = {v: k for k, v in panel2idx.items()}

with open('panelid2name.json', 'r') as f:
    panelid2name = json.load(f)

name2panelid = {v: k for k, v in panelid2name.items()}
# ------------------------------------------
def ensure_gene(gene_symbol: str,
                feat: torch.Tensor | None = None) -> int:
    """
    Make sure `gene_symbol` exists in the graph.
    Returns its internal index (gid).
    """
    gid = gene2idx.get(gene_symbol)
    if gid is None:                                  # brand-new
        gid = len(gene2idx)
        gene2idx[gene_symbol] = gid

        # append a feature row
        if feat is None:           # cheap fallback: small random vector
            feat = torch.randn(data['gene'].x.size(1)) * 0.01
        data['gene'].x = torch.cat([data['gene'].x, feat[None]], 0)
    return gid


def add_edge(gid: int, pid: int):
    """
    Append one (gene gid → panel pid) edge (+ reverse) to the graph.
    """
    e = torch.tensor([[gid], [pid]])
    data['gene', 'in', 'panel'   ].edge_index = torch.cat(
        [data['gene', 'in', 'panel'   ].edge_index, e], dim=1)
    data['panel', 'rev_in', 'gene'].edge_index = torch.cat(
        [data['panel', 'rev_in', 'gene'].edge_index, e.flip(0)], dim=1)


# Simple user inputs
# panel = st.text_area("Enter your node or graph data (JSON):")
panels = st.multiselect("Select a panel",
    options=list(name2panelid.keys()))


if st.button("Predict"):
    try:

        for panel in panels:
            panel = int(name2panelid[panel])
            gene   = 'MYNEWGENE'       # HGNC symbol just entered by a curator

            gid = ensure_gene(panel)              # create/lookup the gene
            pid = panel2idx[panel]               # panels were pre-indexed
            add_edge(gid, pid)                   # store the new membership
            x_dict = {
                'gene' : data['gene'].x,
                'panel': data['panel'].x,
            }
            edge_index_dict = {
                ('gene',  'in',     'panel'): data['gene',  'in',     'panel'].edge_index,
                ('panel', 'rev_in', 'gene' ): data['panel', 'rev_in', 'gene' ].edge_index,
            }
        # ------------------------------------------
        # 4. score every *other* panel for this gene
        # ------------------------------------------
        all_pids  = torch.arange(data['panel'].num_nodes)
        mask      = all_pids != pid              # don’t propose the one we just added
        cand_pids = all_pids[mask]
        pairs     = (torch.full_like(cand_pids, gid), cand_pids)

        with torch.no_grad():
            scores = torch.sigmoid(model(x_dict, edge_index_dict, pairs))

        top = torch.topk(scores, k=10).indices          # best 10
        suggestions = pd.DataFrame({
            "panel_id": [panelid2name.get(str(int(idx2panel[cand_pids[i].item()])), f"Panel {int(idx2panel[cand_pids[i].item()])}") for i in top],
            "score": [round(float(scores[i]), 3) for i in top]
        })
        # suggestions = "".join([f"{str(int(idx2panel[int(cand_pids[i])]))}, {str(round(float(scores[i]),3))}\n" for i in top])
        st.table(suggestions)

        print("Suggested extra panels for", gene, ":\n", suggestions)

    except Exception as e:
        st.error(f"Error: {e}")