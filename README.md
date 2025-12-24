# Sparse Autoencoder for Unsupervised Representation Learning on MNIST

## Project Overview

This repository presents a **from-scratch implementation of a Sparse Autoencoder** using **TensorFlow (Keras)** for **unsupervised representation learning** on the MNIST handwritten digit dataset. The project is designed and written in a manner suitable for a **PhD research portfolio**, emphasizing conceptual clarity, reproducibility, and methodological rigor.

The objective is to learn **low-dimensional, sparse latent representations** of high-dimensional image data while maintaining accurate reconstruction performance. Sparsity is enforced via **L1 activity regularization**, encouraging biologically inspired, efficient neural encodings.

---

## Scientific Motivation

In many real-world domains—such as biomedical imaging, neuromuscular signal processing, and neuroscience—data is high-dimensional but intrinsically low-dimensional. Autoencoders provide a principled unsupervised learning framework to:

- Discover latent structure without labels  
- Compress high-dimensional inputs  
- Learn interpretable and reusable representations  

By enforcing sparsity in the bottleneck layer, this work aligns with theories of **efficient coding** and **sparse neural activation**, which are fundamental in both machine learning and computational neuroscience.

---

## Dataset

- **MNIST Handwritten Digits**
- 70,000 grayscale images
- Image size: 28 × 28
- Flattened input dimension: 784

The dataset is loaded directly using TensorFlow’s built-in utilities.

---

## Data Preparation

- Training set split into:
  - 50,000 samples for training
  - 10,000 samples for validation
- Test set: 10,000 samples
- Images are:
  - Flattened from 28×28 to 784-dimensional vectors
  - Normalized to the range [0, 1]

---

## Model Architecture

The autoencoder consists of a symmetric encoder–decoder structure:

### Encoder
- Input layer: 784 neurons
- Hidden layer: 256 neurons (ReLU activation)
- Bottleneck layer: 32 neurons (ReLU activation)
- L1 activity regularization applied to enforce sparsity

### Decoder
- Hidden layer: 256 neurons (ReLU activation)
- Output layer: 784 neurons (Sigmoid activation)

The bottleneck layer learns a compressed latent representation of the input data.

---

## Training Configuration

- Optimizer: Adam
- Loss function: Binary Cross-Entropy
- Batch size: 256
- Epochs: 50
- Training type: Fully unsupervised (input = target)

---

## Evaluation and Visualization

- Reconstruction loss evaluated on the test set
- Training and validation loss curves plotted for convergence analysis
- Qualitative reconstruction results visualized by comparing:
  - Original MNIST images
  - Autoencoder-reconstructed images

These visual inspections confirm that the sparse latent space preserves essential digit structure.

---

## Key Outcomes

- Successful compression of 784-dimensional inputs into a 32-dimensional sparse latent space
- Stable training with effective reconstruction performance
- Clear demonstration of sparsity-driven representation learning
- Strong foundation for extension to:
  - Biomedical image analysis
  - Signal denoising
  - Feature extraction pipelines
  - Hybrid ML–neuroscience research

---

## Relevance to PhD Research

This project demonstrates:

- Strong understanding of neural network fundamentals
- Practical implementation of regularization techniques
- Clean experimental design and evaluation
- Alignment with research themes in AI, neuroscience, and biomedical engineering

It serves as a **core methodological artifact** suitable for inclusion in a **portfolio** or as a baseline model for more advanced representation learning research.

---

## Technologies Used

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pandas

---

## License & Academic Use

This repository is intended for **academic and research use**. Proper citation is encouraged if the code or methodology is reused in publications or derivative research.

---

## Author

Developed and adapted for research and academic demonstration purposes.  
Prepared as part of a broader **AI + Engineering PhD-oriented portfolio**.
