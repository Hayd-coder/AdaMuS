import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from scipy.stats import wasserstein_distance

# Configuration & Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Dataset Generation
def generate_original_dataset_data():
    CONFIG = {
        'n_samples': 1200,
        'dims': [3000, 10],
        'n_classes': 3,
        'latent_dim': 9,
        'seed': 42
    }
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])

    n_samples = CONFIG['n_samples']
    latent_dim = CONFIG['latent_dim']
    dim_v2 = CONFIG['dims'][1]
    samples_per_class = n_samples // CONFIG['n_classes']

    Z = np.zeros((n_samples, latent_dim))
    for i in range(n_samples):
        base_signal = np.random.gamma(1, 0.8, latent_dim) + np.random.normal(0, 0.1, latent_dim)
        label = i // samples_per_class
        if label == 0:
            Z[i, 0:3] = base_signal[0:3]
        elif label == 1:
            Z[i, 3:6] = base_signal[3:6]
        elif label == 2:
            Z[i, 3:6] = base_signal[3:6]
            Z[i, 6:9] = base_signal[6:9]

    Z_mean = Z.mean(axis=0)
    Z_std = Z.std(axis=0) + 1e-6
    Z_target = (Z - Z_mean) / Z_std

    # W2: View 2 only captures partial latent variables (rows 0-5 are zeroed out)
    W2 = np.random.uniform(0.2, 0.8, (dim_v2, latent_dim))
    W2[:, 0:6] = 0

    noise_v2 = np.random.normal(0, 0.58, (n_samples, dim_v2))
    X2 = np.dot(Z, W2.T) + noise_v2

    return torch.FloatTensor(X2), torch.FloatTensor(Z_target), W2


# 2. Network Definition
class View2Net(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(View2Net, self).__init__()
        # Architecture: 10 -> 128 -> 64 -> 9
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, latent_dim)
        self.msbn = nn.BatchNorm1d(latent_dim, affine=True)  # Sparse BatchNorm Layer

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.relu2(self.bn2(self.fc2(x)))
        feat = self.fc3(x)
        out = self.msbn(feat)
        return out


# 3. Adaptive Pruning Logic
def apply_adaptive_pruning(model, data, device):
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        out = model.fc1(data)
        out = model.bn1(out)
        relu_out = model.relu1(out)

    current_dim = relu_out.shape[1]

    # Calculate pruning rate based on covariance eigenvalues and Wasserstein distance
    relu_out_centered = relu_out - torch.mean(relu_out, dim=0)
    cov_matrix = torch.matmul(relu_out_centered.T, relu_out_centered) / (relu_out.shape[0] - 1)
    eig_value = torch.linalg.eigvalsh(cov_matrix)
    eig_value, _ = torch.sort(eig_value, descending=True)
    eig_value = F.softmax(eig_value, dim=0)

    uni = F.softmax(torch.rand(current_dim), dim=0)
    dirac = torch.zeros(current_dim);
    dirac[0] = 1

    wd_eu = wasserstein_distance(uni.cpu().numpy(), eig_value.cpu().numpy())
    wd_ud = wasserstein_distance(uni.cpu().numpy(), dirac.cpu().numpy())

    prune_rate = min(wd_eu / wd_ud, 0.8)  # Limit max pruning rate

    # Determine kept indices
    mean_activation = torch.mean(relu_out, dim=0)
    keep_num = max(int((1 - prune_rate) * current_dim), 1)
    _, top_indices = torch.topk(mean_activation, k=keep_num, largest=True, sorted=True)
    top_indices, _ = torch.sort(top_indices)

    # Create and apply mask
    mask = torch.zeros(current_dim).to(device)
    mask[top_indices] = 1.0

    with torch.no_grad():
        model.fc1.weight.data *= mask.view(-1, 1)
        model.fc1.bias.data *= mask
        model.bn1.weight.data *= mask
        model.bn1.bias.data *= mask
        model.bn1.running_mean.data *= mask
        model.bn1.running_var.data *= mask
        model.fc2.weight.data *= mask.view(1, -1)

    print(f"Pruning Completed: Rate={prune_rate:.4f}, Retained={keep_num}/{current_dim}")
    return prune_rate, top_indices


# 4. Main Experiment & Plotting
import matplotlib as mpl


def run_experiment_full_logic():
    # 1. Data & Model
    X2, Z_target, gt_W2 = generate_original_dataset_data()
    X2, Z_target = X2.to(DEVICE), Z_target.to(DEVICE)
    model = View2Net(input_dim=10, latent_dim=9).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 2. Training
    EPOCHS = 400
    LAMBDA_L1 = 0.15
    model.train()
    print("Training...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        z_pred = model(X2)
        recon_loss = nn.MSELoss()(z_pred, Z_target)
        sparsity_loss = torch.sum(torch.abs(model.msbn.weight))
        loss = recon_loss + LAMBDA_L1 * sparsity_loss
        loss.backward()
        optimizer.step()

    # 3. Pruning
    apply_adaptive_pruning(model, X2, DEVICE)

    # 4. Prepare data for plotting
    learned_gamma = model.msbn.weight.detach().cpu().numpy()
    gt_map = np.abs(gt_W2.T)

    gamma_viz = np.abs(learned_gamma).reshape(-1, 1)
    gamma_viz = (gamma_viz - gamma_viz.min()) / (gamma_viz.max() - gamma_viz.min() + 1e-8)

    # Plotting Logic
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.1)

    # Left Plot: Ground Truth
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(gt_map, cmap="Greys", cbar=False, vmin=0.05, vmax=0.8,
                linewidths=0.5, linecolor='gray', ax=ax1)

    ax1.set_title("Ground-Truth $W_2^T$", fontsize=18, pad=12, fontweight='medium')
    ax1.set_ylabel("Latent Dimension Index", fontsize=14, fontweight='medium')
    ax1.set_xlabel("Input Feature Index", fontsize=14, fontweight='medium')

    rect1 = patches.Rectangle((0, 0), 10, 6, linewidth=2.5, edgecolor='red', facecolor='none', linestyle='--')
    ax1.add_patch(rect1)

    # Right Plot: Learned Gamma
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(gamma_viz, cmap="Greys", cbar=False, vmin=0.05, vmax=0.8,
                linewidths=0.5, linecolor='gray', ax=ax2, annot=True, fmt=".2f",
                annot_kws={"size": 12, "weight": "bold"})

    ax2.set_title("Learned $\gamma$", fontsize=18, pad=12, fontweight='medium')

    ax2.set_xticks([])
    ax2.set_yticks([])

    rect2 = patches.Rectangle((0, 0), 1, 6, linewidth=2.5, edgecolor='red', facecolor='none', linestyle='--')
    ax2.add_patch(rect2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment_full_logic()