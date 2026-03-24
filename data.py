import pickle
import scipy.io
import scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding


class MultiViewDataset(Dataset):
    def __init__(self, data_path, train=True, custom_views=None):
        super().__init__()

        # Data loading and preprocessing
        dataset = scipy.io.loadmat(data_path)
        self.x = dict()

        # Logic for processing Mfeat dataset (using specific keys)
        if 'handwritten' in data_path or 'Mfeat' in data_path:
            # Define the 6 standard Mfeat view keys
            view_keys = ['mfeat_fac', 'mfeat_fou', 'mfeat_kar', 'mfeat_mor', 'mfeat_pix', 'mfeat_zer']
            for i, key in enumerate(view_keys):
                if key in dataset:
                    view_data = dataset[key]
                    self.x[i] = MinMaxScaler((0, 1)).fit_transform(view_data).astype(np.float32)
                else:
                    raise KeyError(f"View key not found in Mfeat.mat: '{key}'")

            # Key Step: Programmatically generate labels
            # Mfeat dataset contains 10 classes, 200 samples each, arranged sequentially
            num_samples = self.x[0].shape[0] # Usually 2000
            if num_samples != 2000:
                 print(f"Warning: Expected 2000 samples, got {num_samples}")

            num_classes = 10
            samples_per_class = num_samples // num_classes

            self.y = np.array([i // samples_per_class for i in range(num_samples)], dtype=np.int64)
            print(f"Note: Mfeat dataset doesn't have labels. Generated labels programmatically based on structure ({num_classes} classes, {samples_per_class} per class).")

        elif '100Leaves' in data_path:
            # Compatibility for 100Leaves.mat: usually contains 'X' (cell array) and 'Y' (labels)
            if 'X' in dataset and 'Y' in dataset:
                print("Detected 100Leaves dataset, loading data from cell array structure...")
                data_container = dataset['X']
                # X is usually a 1 x V (views) cell array
                num_views = data_container.shape[1] if data_container.shape[0] == 1 else data_container.shape[0]
                
                for k in range(num_views):
                    # Assuming 1 x V
                    if data_container.shape[0] == 1:
                        view_data = data_container[0, k] 
                    else:
                        view_data = data_container[k, 0]

                    # check if the element is sparse or strict
                    if sp.issparse(view_data):
                        print(f"View {k + 1} is a sparse matrix, using MaxAbsScaler.")
                        # fit_transform might return sparse matrix or dense depending on implementation
                        # MaxAbsScaler supports sparse input
                        self.x[k] = MaxAbsScaler().fit_transform(view_data).toarray().astype(np.float32)
                    else:
                        self.x[k] = MinMaxScaler((0, 1)).fit_transform(view_data).astype(np.float32)
                    
                    print(f"View {k + 1} loaded, shape: {self.x[k].shape}")
                
                self.y = dataset['Y'].flatten().astype(np.int64)
                print(f"Labels loaded, total samples: {len(self.y)}")
            else:
                 raise KeyError("'X' or 'Y' field not found in 100Leaves.mat")

        else:
            raise ValueError(f"This code currently does not support the dataset path: {data_path}")

        # Common cleanup logic
        if min(self.y) > 0:
            self.y -= 1

        if custom_views is not None:
            self.x = {k: self.x[k] for k in custom_views}

    def __getitem__(self, index):
        x = dict()
        for v in self.x.keys():
            x[v] = self.x[v][index]
        return {
            'x': x,
            'y': self.y[index],
            'index': index
        }

    def __len__(self):
        return len(self.y)

