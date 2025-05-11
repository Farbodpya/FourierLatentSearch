# FourierLatentSearch

FourierLatentSearch is an optimization framework that applies a Genetic Algorithm (GA) in a **latent space** transformed via a **Fourier-based projection**. This technique helps reduce the dimensionality of high-dimensional optimization problems while preserving useful structure for search.

Zenodo DOI: (https://doi.org/10.5281/zenodo.15342205)

---

## 🔬 Core Idea

High-dimensional optimization is computationally expensive and prone to getting stuck in local minima. To address this, **FourierLatentSearch**:

1. **Projects** the high-dimensional search space into a low-dimensional latent space using **randomized Fourier projection**.
2. **Optimizes** in this lower-dimensional space using a **Genetic Algorithm** (GA) with dynamic population size.
3. **Decodes** latent vectors back into the original high-dimensional space for evaluation on benchmark functions.

The Fourier transform helps to capture global structure in the original space using frequency components, offering an effective form of latent representation.

---

## 🧠 Features

- **Random Fourier Projection** for latent transformation.
- Dynamic population size GA with blend crossover and adaptive mutation.
- Supports classic benchmark functions like Sphere, Rastrigin, Ackley, Griewank, Zakharov, and a custom variant: Rastrigin II.
- Modular code structure for easy experimentation.

---

## 📊 Benchmarks

Functions supported:

- Sphere
- Rastrigin
- Rastrigin II
- Ackley
- Griewank
- Zakharov

Each function is evaluated in the high-dimensional space (default: 5000D), with GA operating in a reduced latent space (e.g., 2D).

---

## 🚀 Usage

1. **Clone the repo**:

   ```bash
   git clone https://github.com/Farbodpya/FourierLatentSearch.git
   cd FourierLatentSearch

   # Run the optimizer
   python main.py
