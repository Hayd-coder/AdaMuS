import os
os.environ["OMP_NUM_THREADS"] = "1"
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import MultiViewDataset
from model import AdaMus
from cluster import cluster
from tsne import my_tsne
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def train(model, train_num, train_data, train_loader, valid_loader, epochs: list, k, n_clusters, save_weights_to, w_path, device):
    model = model.to(device)
    x_tensor = {}
    for i in train_data.x.keys():
        x_tensor[i] = torch.tensor(train_data.x[i]).to(device)

    # Freeze S and A during fine-tuning
    model.S.requires_grad_(False)
    model.A.requires_grad_(False)
    
    optimizer2 = torch.optim.Adam([
        {'params': (p for n, p in model.named_parameters()  # Group 1: Apply L2 regularization to weights
                    if p.requires_grad and 'weight' in n), 'weight_decay': 0.01},
        {'params': (p for n, p in model.named_parameters()  # Group 2: Other parameters without weight_decay
                    if p.requires_grad and 'weight' not in n)},
    ], lr=0.01)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode='max', factor=0.1, min_lr=1e-7, patience=5, verbose=True, threshold=0.0001, eps=1e-08
    )
    
    best_model_wts = model.state_dict()
    best_acc = 0

    for epoch in range(epochs[2]):
        model.train()
        train_loss, num_samples = 0, 0
        
        for batch in train_loader:
            x, y, index = batch['x'], batch['y'], batch['index']
            for v in x.keys():
                x[v] = x[v].to(device)
            index = index.to(device)
            
            # Get similarity S between samples in the current batch and all other samples (batch x train_num)
            batch_s = model.S[index, :]    
            # Extract relationships among the batch samples (batch x batch)
            batch_s = batch_s[:, index]    
            
            ret = model(x, S=batch_s, batch=len(y)) 
            optimizer2.zero_grad()
            
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                
            ret['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer2.step()
            
            train_loss += ret['loss'].mean().item() * len(y) 
            num_samples += len(y)

        train_loss = train_loss / num_samples
        
        ret = model.fusion_z(x_tensor)
        fusion = ret['fusion'].data.cpu().numpy()
        
        print("*******************************")
        acc, _, nmi, _, ri, _, f1, _ = cluster(n_clusters, fusion, train_data.y, count=5)
        scheduler.step(acc)
        
        if best_acc < acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
        print(f'Epoch {epoch:3d}: train loss {train_loss:.4f}, acc {acc/100:.6f}, nmi {nmi/100:.4f}, ri {ri/100:.4f}')

    if save_weights_to is not None:
        os.makedirs(os.path.dirname(save_weights_to), exist_ok=True)
        torch.save(best_model_wts, save_weights_to)

    print("\n--- Fine-tuning complete. Calculating final performance with the best model... ---")
    model.load_state_dict(best_model_wts)

    # Recalculate final performance metrics using the best model
    ret = model.fusion_z(x_tensor)
    fusion = ret['fusion'].data.cpu().numpy()
    
    # Set count to 10 for more stable evaluation results
    acc, _, nmi, _, ari, _, f1, _ = cluster(n_clusters, fusion, train_data.y, count=10) 
    
    print(f"Final Performance: ACC={acc:.2f}, NMI={nmi:.2f}, ARI={ari:.2f}, F1={f1:.2f}\n")

    return model, acc, nmi, ari, f1


def validate(model, loader, n_clusters, device='cuda'):
    model.eval()
    with torch.no_grad():
        acc, nmi, ri, f1, num_samples = 0, 0, 0, 0, 0
        for batch in loader:
            x = batch['x']
            y = batch['y']
            for k in x.keys():
                x[k] = x[k].to(device)
            y = y.to(device)
            
            ret = model.fusion_z(x)  
            acc_avg, _, nmi_avg, _, ri_avg, _, f1_avg, _ = cluster(n_clusters, ret['fusion'], y, count=5)
            
            acc += acc_avg
            nmi += nmi_avg
            ri += ri_avg
            f1 += f1_avg
            num_samples += len(batch['y'])
            
        print(f'acc {acc/num_samples:.4f}, nmi {nmi/num_samples:.4f}, ri {ri/num_samples:.4f}, f1 {f1/num_samples:.4f}')
    return


def experiment(data_path, com_dim, en_dim, low_dim, low_index, high_index, high_dim, epochs: list, k, device, para_path, best_weight, w_path, lambda1, margin_value):
    train_data = MultiViewDataset(data_path=data_path, train=True)
    train_num = len(train_data)
    
    if train_num > 5000:
        batch_size = 256
        print(f"Large sample size ({train_num}), automatically setting batch_size to {batch_size}")
    elif train_num > 1500:
        batch_size = 128
        print(f"Medium sample size ({train_num}), automatically setting batch_size to {batch_size}")
    else:
        batch_size = 64
        
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    dim_list = []
    sample_shape = [s.shape[0] for s in train_data[0]['x'].values()] 
    
    for i in range(len(sample_shape)):
        if i == low_index:
            dim_list.append([sample_shape[i], low_dim])
        elif i in high_index:
            dim_list.append([high_dim, en_dim])
        else:
            dim_list.append([sample_shape[i], en_dim])

    print('---------------------------- Experiment ------------------------------')
    print('Dataset:', data_path)
    print('Number of views:', len(train_data.x), ' views with dims:', [v.shape[-1] for v in train_data.x.values()])
    print('Number of classes:', len(set(train_data.y)))
    print('Number of training samples:', len(train_data))
    print('Trainable Parameters:')
    
    print("---------------------- Pre-training ------------------")
    x_tensor = {}
    for i in train_data.x.keys():
        x_tensor[i] = torch.tensor(train_data.x[i]).to(device)

    prune_model = AdaMus(
        sample_shape=sample_shape,
        train_num=train_num,
        com_dim=com_dim,
        high_index=high_index,
        high_dim=high_dim,
        dim_list=dim_list, 
        device=device,
        lambda1=lambda1,
        margin_value=margin_value
    )
    
    # Pre-training: only updates graph structure S and weights A
    prune_model.pre_train(
        train_num, x_tensor, train_data.y, train_loader, valid_loader=None, 
        epochs=epochs, k=k, n_clusters=len(set(train_data.y)), w_path=w_path, device=device
    )

    print("------------------------- Pruning --------------------------") 
    # Prune and save the retained network parameters to para_path
    prune_dim_list = prune_model.prune_view(x_tensor, dim_list, para_path, device) 

    # Recreate the model instance to load the pruned network parameters
    umdl = AdaMus(
        sample_shape=sample_shape,
        train_num=train_num,
        com_dim=com_dim,
        high_index=high_index,
        high_dim=high_dim,
        dim_list=prune_dim_list, # Network structure after pruning
        device=device,
        lambda1=lambda1,
        margin_value=margin_value
    )
    
    umdl.load_state_dict(torch.load(para_path))

    for n, p in umdl.named_parameters():
        print('%-40s' % n, '\t', p.data.shape)
        
    print('----------------------------- Fine-tuning -----------------------------------------')
    _, acc, nmi, ari, f1 = train(
        model=umdl, train_num=train_num, train_data=train_data, train_loader=train_loader,
        valid_loader=None, epochs=epochs, k=k, n_clusters=len(set(train_data.y)), 
        save_weights_to=best_weight, w_path=w_path, device=device
    )

    return acc, None, nmi, None, ari, None, f1, None


if __name__ == '__main__':
    # epochs format: [graph, prune_train, pre_train]
    experiment(
        data_path="data/100Leaves.mat",
        com_dim=128,
        en_dim=256,
        low_dim=64,
        low_index=-1,             # -1 indicates no specifically assigned low-dimensional view
        high_index=[],
        high_dim=512,
        epochs=[50, 20, 30],
        k=30,                     # Slightly larger k value suitable for a sample size of 1600
        device="cuda",
        para_path="prune_para/100leaves_para.pth",
        best_weight="best_para/100leaves_best.pth",
        w_path="w_para/100leaves_w.pth",
        lambda1=0.1,
        margin_value=0.05
    )