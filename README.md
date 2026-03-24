# AdaMuS: Adaptive Multi-view Sparsity Learning for Dimensionally Unbalanced Data

This repository contains the official implementation of **AdaMuS** from the paper:  
**"[AdaMuS: Adaptive Multi-view Sparsity Learning for Dimensionally Unbalanced Data](https://arxiv.org/abs/2603.17610)"** (arXiv:2603.17610).

The framework implements the **AdaMuS** algorithm for dimensionally unbalanced data.

## Abstract

Multi-view learning primarily aims to fuse multiple features to describe data comprehensively. Most prior studies implicitly assume that different views share similar dimensions. In practice, however, severe dimensional disparities often exist among different views, leading to the unbalanced multi-view learning issue. For example, in emotion recognition tasks, video frames often reach dimensions of , while physiological signals comprise only dimensions. Existing methods typically face two main challenges for this problem: (1) They often bias towards high-dimensional data, overlooking the low-dimensional views. (2) They struggle to effectively align representations under extreme dimensional imbalance, which introduces severe redundancy into the low-dimensional ones. To address these issues, we propose the Adaptive Multi-view Sparsity Learning (AdaMuS) framework. First, to prevent ignoring the information of low-dimensional views, we construct view-specific encoders to map them into a unified dimensional space. Given that mapping low-dimensional data to a high-dimensional space often causes severe overfitting, we design a parameter-free pruning method to adaptively remove redundant parameters in the encoders. Furthermore, we propose a sparse fusion paradigm that flexibly suppresses redundant dimensions and effectively aligns each view. Additionally, to learn representations with stronger generalization, we propose a self-supervised learning paradigm that obtains supervision information by constructing similarity graphs. Extensive evaluations on a synthetic toy dataset and seven real-world benchmarks demonstrate that AdaMuS consistently achieves superior performance and exhibits strong generalization across both classification and semantic segmentation tasks.

## Dataset

The project is pre-configured for the **100Leaves** dataset (`100Leaves.mat`), located in the `data/` directory.

- **Classes**: 100 (Plant Species)
- **Samples**: 1600 (16 per class)
- **Views**: 3 (Shape, Texture, Margin) - each with dimension 64.

_Note: The dataset loader automatically handles the 100Leaves structure._

## Prerequisites

- Python 3.x
- PyTorch
- NumPy
- SciPy
- Scikit-learn
- Matplotlib

You can install the required packages using pip:

```bash
pip install torch numpy scipy scikit-learn matplotlib
```

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Hayd-coder/AdaMus_Pruning.git
   cd AdaMus_Pruning
   ```

2. **Run the training script:**

   ```bash
   python main.py
   ```

3. **Output:**
   - The script will print clustering metrics (ACC, NMI, ARI, F1) to the console during training.
   - Model checkpoints and parameters are saved in:
     - `best_para/`: Best model weights.
     - `prune_para/`: Pruning parameters.
     - `w_para/`: View weights.

## Project Structure

- `main.py`: Main entry point for training and evaluation.
- `model.py`: Defines the `ProposedModel` architecture and the pruning logic.
- `data.py`: Handles dataset loading and preprocessing for Mfeat.
- `cluster.py`: Implements K-Means clustering and metric calculation.
- `tsne.py`: t-SNE visualization utility.
- `data/`: Contains the dataset file 

## License

[MIT License](LICENSE)
