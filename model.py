from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from cluster import cluster
from scipy.stats import wasserstein_distance
import copy


class ParallelModule(nn.Module):
    def __init__(self, modules: Tuple[nn.Module]):
        super().__init__()
        self.parallel_modules = nn.ModuleList(modules)

    def forward(self, x: dict):
        y = dict()
        for k, v in x.items():
            # Returned in order; k corresponds to the index
            y[k] = self.parallel_modules[k](v)  
        return y


class AdaMus(nn.Module):
    def __init__(self, sample_shape: list, train_num, com_dim, high_index, high_dim, dim_list, device, lambda1, margin_value):
        super().__init__()
        self.sample_shape = sample_shape  # Dimension size of each view
        self.view_num = len(sample_shape)  # Number of views
        self.high_index = high_index  # Indices of high-dimensional views
        self.com_dim = com_dim  # Size of the common dimension
        self.dim_list = dim_list  # Structural information for each view (input -> hidden layer)
        self.high_sample_shape = [sample_shape[i] for i in high_index]

        self.lambda1 = lambda1
        self.device = device
        self.margin = torch.tensor(margin_value)

        # S: Consensus matrix, W: Similarity matrix for each view
        self.W = dict()  
        self.S = nn.Parameter(torch.rand(train_num, train_num))  
        self.A = nn.Parameter(torch.rand(self.view_num))  # Weight coefficient for each view

        # Apply a Linear+ReLU network to each high-dimensional view to project them into the same high_dim space
        self.rd_nets = ParallelModule([nn.Sequential(
            nn.Linear(shape, high_dim),
            nn.ReLU()
        ) for shape in self.high_sample_shape])  

        self.layer = [nn.Sequential(
            nn.Linear(pair[0], pair[1]),
            nn.BatchNorm1d(pair[1]),
            nn.ReLU(),
            nn.Linear(pair[1], com_dim),
            nn.BatchNorm1d(com_dim),
            nn.ReLU()
        ) for pair in dim_list]

        # Encoders project data into an aligned space; each sub-network is an independent layer
        self.encoder = ParallelModule(self.layer)  

    def similar_matrix(self, X, k):  
        # Build a similarity matrix for each view
        # X shape: (n, d)
        num = X.shape[0]
        
        # Sum of squares for each sample, shape (n, 1)
        sqr_x = torch.sum(X ** 2, dim=1, keepdim=True)  
        # Sum of squared Euclidean distances, shape (n, n)
        dist = sqr_x + sqr_x.t() - 2 * (X @ X.t())  
        
        # Find the k nearest neighbors for each sample
        knn = dict()
        sortx, index = torch.sort(dist, dim=1, descending=False)  
        for i in range(num):
            knn[i] = index[i][1:k + 1]  

        # Construct the similarity matrix
        W = torch.eye(num, num)  
        sigma = 0.0
        for i in range(num):
            sigma += sortx[i][k + 1]
        sigma = sigma / num
        sigma = 2 * sigma.pow(2)

        for i in range(num):
            for j in range(i, num):  
                if knn[i].tolist().count(j) > 0 or knn[j].tolist().count(i) > 0:
                    if W[i][j] <= 0:
                        # Calculate similarity; smaller distance yields higher similarity
                        W[i][j] = torch.exp(-dist[i][j] / sigma)  
                    if W[j][i] <= 0:
                        W[j][i] = W[i][j]  # Construct symmetric matrix W
        return W

    def forward(self, x: dict, S, batch):  
        # Input x is a dictionary, S is the consensus matrix of shape [batch, batch]
        if len(self.high_index) != 0:
            view = dict()  
            re_x = {}
            for i, num in enumerate(self.high_index):
                re_x[i] = x[num]  # Extract high-dimensional data into re_x
                
            # Feed high-dimensional data into their respective networks for dimensionality reduction
            rd_output = self.rd_nets(re_x)  
            
            for i in range(self.view_num):
                if i in self.high_index:
                    view[i] = rd_output[self.high_index.index(i)]  
                else:
                    view[i] = x[i]  # Assign low-dimensional data directly
                    
            # Pass view into the encoder to project into the aligned space
            zn_v = self.encoder(view)  
        else:
            zn_v = self.encoder(x)  

        fusion = sum(zn_v.values()) / self.view_num
        ret = {'fusion': fusion}

        # Multi-view batch normalization
        gama_l1norm = 0
        for k in range(self.view_num):  
            last_bn = None
            for net in self.encoder.parallel_modules[k]:
                if isinstance(net, nn.BatchNorm1d):
                    last_bn = net  # Find the last bn layer
            # Calculate L1 regularization term
            gama_l1norm += last_bn.weight.abs().mean()  

        # Fused graph reconstruction error
        sqr_fusion = torch.sum(fusion ** 2, dim=1, keepdim=True)
        dist_sqr = sqr_fusion + sqr_fusion.t() - 2 * (fusion @ fusion.t())
        
        # Squared distances between samples in the aligned space for the current batch
        dist_sqr = dist_sqr / batch ** 2  
        dist_sqr.to(self.device)

        A = torch.zeros((batch, batch), dtype=torch.bool).to(self.device)
        
        # Define positive and negative samples
        for i in range(batch):
            for j in range(batch):  
                if S[i][j] > 0.60 and A[i][j] == False:
                    A[i][j], A[j][i] = True, True
                    
        cons_loss = torch.tensor(0, dtype=torch.float32).to(self.device)
        zero = torch.tensor(0, dtype=torch.float).to(self.device)

        for i in range(batch):
            for j in range(batch):
                if i != j:
                    if A[i][j] == True:
                        cons_loss += dist_sqr[i][j]
                        cons_loss.to(self.device)
                    else:
                        cons_loss += torch.max(
                            self.margin - torch.sqrt(dist_sqr[i][j]),
                            zero).to(self.device) ** 2

        cons_loss = cons_loss / 2  
        ret['loss'] = cons_loss + self.lambda1 * gama_l1norm
        return ret  

    def fusion_z(self, x):
        if len(self.high_index) != 0:  
            view = dict()  
            re_x = {}
            for i, num in enumerate(self.high_index):
                re_x[i] = x[num]
            rd_output = self.rd_nets(re_x)
            
            for i in range(self.view_num):
                if i in self.high_index:
                    view[i] = rd_output[self.high_index.index(i)]  
                else:
                    view[i] = x[i]  
            zn_v = self.encoder(view)  
        else:
            zn_v = self.encoder(x)
            
        fusion = sum(zn_v.values()) / self.view_num
        ret = {'fusion': fusion}
        return ret

    def pre_train(self, train_num, train_data_x, trian_data_y, train_loader, valid_loader, epochs, k, n_clusters, w_path, device):
        self = self.to(device)

        optimizer1 = torch.optim.Adam([
            {'params': self.S},
            {'params': self.A},
        ], lr=0.01)

        if os.path.exists(w_path):
            self.W = torch.load(w_path)
            if device == "cpu":
                for i in self.W.keys():
                    self.W[i] = self.W[i].to(device)
        else:
            # Build similarity matrix for all train data
            for i in range(self.view_num):
                print("build similarity matrix " + str(i) + ":---")
                self.W[i] = self.similar_matrix(train_data_x[i], k).to(device)
            if w_path is not None:
                os.makedirs(os.path.dirname(w_path), exist_ok=True)
                torch.save(self.W, w_path)

        # Train to find optimal S and A
        print("similarity matrix training:------------------------")
        for epoch in range(epochs[0]):  
            rec = torch.tensor(0).to(device)
            for i in range(self.view_num):
                # Weighted fusion of similarity matrices from all views
                rec = rec + self.A[i] * self.W[i]  
                
            loss = torch.dist(self.S, rec, 2) ** 2 / train_num ** 2  
            optimizer1.zero_grad()
            loss.backward(retain_graph=True)
            optimizer1.step()
            self.A.data = nn.Softmax(dim=0)(self.A.data)
            print('epoch{}---:, loss{}'.format(epoch, loss))

        # Forward pass preparation
        self.S.requires_grad_(False)  # Freeze S and A to prevent gradient updates
        self.A.requires_grad_(False)
        
        optimizer2 = torch.optim.Adam([
            {'params': (p for n, p in self.named_parameters()
                        if p.requires_grad and 'weight' in n), 'weight_decay': 0.01},
            {'params': (p for n, p in self.named_parameters()
                        if p.requires_grad and 'weight' not in n)},
        ], lr=0.01)
        
        # Configure scheduler to reduce learning rate if there's no improvement
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer2, mode='max', factor=0.5, min_lr=1e-7, patience=4, verbose=True, threshold=0.0001, eps=1e-08
        )  
        
        for epoch in range(epochs[1]):
            self.train()
            train_loss, num_samples = 0, 0
            for batch in train_loader:
                x, y, index = batch['x'], batch['y'], batch['index']
                for k in x.keys():
                    x[k] = x[k].to(device)
                index = index.to(device)
                
                # Obtain the consensus matrix for the current batch
                batch_s = self.S[index, :]
                batch_s = batch_s[:, index]  
                
                ret = self(x, S=batch_s, batch=len(y))  
                optimizer2.zero_grad()

                ret['loss'].backward()  
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer2.step()
                train_loss += ret['loss'].mean().item() * len(y)  
                num_samples += len(y)

            train_loss = train_loss / num_samples
            ret = self.fusion_z(train_data_x)
            fusion = ret['fusion'].data.cpu().numpy()
            print("*******************************")
            acc, _, nmi, _, ri, _, f1, _ = cluster(n_clusters, fusion, trian_data_y, count=5)
            scheduler.step(acc)
            print(f'Epoch {epoch:3d}: train loss {train_loss:.4f}, acc {acc / 100:.6f}, nmi {nmi / 100:.4f}, ri {ri / 100:.4f}')

    def prune_view(self, x, dim_list, para_path, device):
        # Generate a uniform distribution and a Dirac distribution for each layer, and calculate their Wasserstein distance
        uni = []
        dirac = []
        wd_ud = []
        for i in range(len(dim_list)):  
            uni.append(torch.rand(dim_list[i][1]))  
            dirac.append(torch.zeros(dim_list[i][1]))
            dirac[i][1] = 1
            wd_ud.append(wasserstein_distance(uni[i], dirac[i]))

        # Calculate the adjustment factor tau^v according to equations (16a) and (16b)
        
        # 1. Calculate D_v, D_bar, sigma_D
        all_dims = torch.tensor(self.sample_shape, dtype=torch.float32).to(device)
        d_bar = torch.mean(all_dims)
        sigma_d = torch.std(all_dims)
        
        # Prevent division by zero if all dimensions are identical
        if sigma_d < 1e-8:
            sigma_d = 1e-8

        # 2. Calculate p^v, p_bar, sigma_p
        # p^v is the ratio between the common dimension com_dim and the original dimension D_v
        p_all = torch.tensor([self.com_dim / d for d in self.sample_shape], dtype=torch.float32).to(device)
        p_bar = torch.mean(p_all)
        sigma_p = torch.std(p_all)
        
        # Prevent division by zero if all ratios are identical
        if sigma_p < 1e-8:
            sigma_p = 1e-8

        # 3. Calculate Lambda^v and tau^v for each view
        tau_list = []
        for i in range(self.view_num):
            D_v = all_dims[i]
            p_v = p_all[i]
            # Calculate imbalance degree Lambda^v according to equation (16b)
            Lambda_v = (torch.abs(D_v - d_bar) / (2 * sigma_d)) + (torch.abs(p_v - p_bar) / (2 * sigma_p))
            # Calculate adjustment factor tau^v according to equation (16a)
            tau_v = 2 / (1 + torch.exp(Lambda_v))
            tau_list.append(tau_v)

        print("Calculated adjustment factors tau for each view:", [f'{t.item():.4f}' for t in tau_list])

        prune_rate = []
        index_list = []
        return_list = copy.deepcopy(self.dim_list)
        for i in range(len(dim_list)):
            relu_out = x[i]
            for net in self.layer[i]:  
                relu_out = net(relu_out)  
                if isinstance(net, nn.ReLU):
                    # Covariance matrix calculation
                    relu_out = relu_out - torch.mean(relu_out, axis=0)
                    cov_matrix = torch.matmul(relu_out.T, relu_out) / (relu_out.shape[0] - 1)  
                    
                    # Compute eigenvalues, returning a complex tensor
                    eig_complex = torch.linalg.eigvals(cov_matrix)  
                    eig = eig_complex.real  # Take the real part to get a real tensor

                    max_value = torch.max(eig)
                    min_value = torch.min(eig)
                    # Normalize all eigenvalues to the [0, 1] range
                    eig_value = (eig - min_value) / (max_value - min_value)  

                    # Calculate W(lambda, u)
                    wd_eu = wasserstein_distance(
                        uni[i].cpu().detach().numpy(), eig_value.cpu().detach().numpy())

                    # Calculate the final prune rate gamma^v_l according to equation (15)
                    base_prune_rate = wd_eu / wd_ud[i]
                    tau_v = tau_list[i].item()
                    final_prune_rate = base_prune_rate * tau_v
                    prune_rate.append(base_prune_rate)

                    # Calculate the pruned dimension using the adjusted prune rate
                    low_dim = int((1 - final_prune_rate) * dim_list[i][1])  
                    return_list[i][1] = low_dim
                    
                    # Calculate the reciprocal of the standard deviation for each column
                    reciprocal_var = 1 / torch.sqrt(torch.diag(cov_matrix))
                    
                    # Calculate Pearson Correlation Coefficient matrix
                    pcc = cov_matrix * reciprocal_var * reciprocal_var.view(dim_list[i][1], 1)  
                    
                    # L1-norm calculation
                    l1_norm = torch.norm(pcc, p=1, dim=1)
                    
                    # Keep small l1_norm
                    _, index = torch.topk(l1_norm, k=low_dim, largest=False, sorted=True)
                    index, _ = torch.sort(index)
                    index_list.append(index)
                    break

        para_dict = self.state_dict()
        
        for i in range(len(dim_list)):
            # Iterate through all views and replace the original weight matrix with the retained neurons' weights for dimensionality reduction
            para_dict[f'encoder.parallel_modules.{i}.0.weight'] = para_dict[f'encoder.parallel_modules.{i}.0.weight'][index_list[i], :]
            para_dict[f'encoder.parallel_modules.{i}.0.bias'] = para_dict[f'encoder.parallel_modules.{i}.0.bias'][index_list[i]]
            para_dict[f'encoder.parallel_modules.{i}.1.weight'] = para_dict[f'encoder.parallel_modules.{i}.1.weight'][index_list[i]]
            para_dict[f'encoder.parallel_modules.{i}.1.bias'] = para_dict[f'encoder.parallel_modules.{i}.1.bias'][index_list[i]]
            para_dict[f'encoder.parallel_modules.{i}.1.running_mean'] = para_dict[f'encoder.parallel_modules.{i}.1.running_mean'][index_list[i]]
            para_dict[f'encoder.parallel_modules.{i}.1.running_var'] = para_dict[f'encoder.parallel_modules.{i}.1.running_var'][index_list[i]]

            # Layer 2 adjustments
            para_dict[f'encoder.parallel_modules.{i}.3.weight'] = para_dict[f'encoder.parallel_modules.{i}.3.weight'][:, index_list[i]]

        # Save model
        if para_path is not None:
            os.makedirs(os.path.dirname(para_path), exist_ok=True)
            torch.save(para_dict, para_path)
            
        # Print information
        print("Pruning completed-------------------------------")
        print("Original network dimensions: ", self.dim_list)
        print("Pruning rates: ", prune_rate)
        print("Dimensions after pruning: ", return_list)

        return return_list