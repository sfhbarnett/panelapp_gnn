import torch
from torch_geometric.nn import HeteroConv, SAGEConv

class GenePanelGNN(torch.nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.conv1 = HeteroConv({
            ('gene','in','panel'):  SAGEConv((-1, -1), hidden),
            ('panel','rev_in','gene'): SAGEConv((-1, -1), hidden)
        }, aggr='mean')
        self.conv2 = HeteroConv({
            ('gene','in','panel'):  SAGEConv((-1, -1), hidden),
            ('panel','rev_in','gene'): SAGEConv((-1, -1), hidden)
        }, aggr='mean')
        self.score = torch.nn.Bilinear(hidden, hidden, 1)   # link predictor

    def forward(self, x_dict, edge_index_dict, pairs):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: x.relu() for k,x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        g_emb = x_dict['gene'][pairs[0]]
        p_emb = x_dict['panel'][pairs[1]]
        return self.score(g_emb, p_emb).squeeze(-1)
    
def load_model(path):
    model = GenePanelGNN()  # supply init args
    model.load_state_dict(torch.load("gene_panel_gnn.pth"))
    model.eval()
    return model

def predict(model, input_data):
    # Convert input_data to tensor
    # Run inference
    return {"result": "your output here"}
