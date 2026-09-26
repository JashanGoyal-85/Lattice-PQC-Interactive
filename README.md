# Lattice-Based Post-Quantum Cryptography — Interactive Implementation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JashanGoyal-85/Lattice-PQC-Interactive/blob/main/Lattice_PQC_Colab_Notebook.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An interactive Python implementation and visualization of **lattice-based post-quantum cryptography (PQC)** concepts, with **machine learning experiments** exploring the intersection of ML and PQC security.

> Based on the paper: *"An interactive tool for teaching lattice-based post-quantum cryptography"*
> — Pérez-Ramos, Hernández-Goya & Caballero-Gil (2026), Journal of New Approaches in Educational Research, 15:18.
> [DOI: 10.1007/s44322-026-00068-x](https://doi.org/10.1007/s44322-026-00068-x)

---

## 📌 Overview

Quantum computers threaten current cryptographic standards (RSA, ECDSA). **Lattice-based cryptography** is the leading replacement — already standardized by NIST as **ML-KEM (FIPS 203)** and **ML-DSA (FIPS 204)**. This project provides a hands-on, visual, and ML-enhanced exploration of the mathematical foundations behind these algorithms.

---

## 🧪 What's Implemented

### Core Cryptographic Concepts

| # | Topic | Description |
|---|-------|-------------|
| 1 | **Lattice Generation** | 2D lattice visualization with basis vectors and fundamental domains |
| 2 | **Dot Product & Orthogonality** | How basis orthogonality affects security — 4-basis comparison |
| 3 | **Euclidean vs. Manhattan Distance** | Taxi distance on lattice grids (Barcelona example from the paper) |
| 4 | **Closest Vector Problem (CVP)** | Visual demo with distance circles and closest-point identification |
| 5 | **Shortest Vector Problem (SVP)** | SVP radius visualization; shortest vector ≠ basis vector |
| 6 | **LWE Encryption (Regev Scheme)** | Full key-gen → encrypt → decrypt pipeline for bit-level messages |
| 7 | **CRYSTALS-Kyber Key Exchange** | Simplified ML-KEM: Alice & Bob establish a shared secret |
| 8 | **Dimension vs. Difficulty** | Empirical proof that lattice problems scale exponentially |

### Machine Learning + PQC Experiments

| # | Experiment | Key Result |
|---|-----------|------------|
| 9 | **Neural Network LWE Attack** | MLP trained to crack LWE — **fails at dim ≥ 16** (drops to random guess) |
| 10 | **ML-Assisted CVP Solver** | Neural regression for CVP — accuracy degrades sharply with dimension |
| 11 | **Encrypted ML Inference (FHE)** | Privacy-preserving prediction: cloud computes on ciphertext, never sees raw data |
| 12 | **ML vs. Security Scaling** | Combined visualization: ML accuracy, search space, NIST security levels |

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Click the **"Open in Colab"** badge above to launch directly.
2. Or manually: go to [colab.research.google.com](https://colab.research.google.com) → **Upload** → select `Lattice_PQC_Colab_Notebook.ipynb`
3. Run all cells with **Runtime → Run All**

### Option 2: Local Execution

```bash
# Clone the repository
git clone https://github.com/JashanGoyal-85/Lattice-PQC-Interactive.git
cd Lattice-PQC-Interactive

# Install dependencies
pip install -r requirements.txt

# Run the notebook
python Lattice_PQC_Colab_Notebook.py
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.21 | Linear algebra, lattice operations |
| `matplotlib` | ≥ 3.5 | Visualizations and plots |
| `scikit-learn` | ≥ 1.0 | ML experiments (MLP classifier/regressor) |

> All three are **pre-installed in Google Colab** — no setup needed!

---

## 📊 Sample Outputs

### Lattice Visualization (Orthogonal vs. Non-Orthogonal)
The notebook generates side-by-side comparisons showing how basis vector choice affects lattice geometry and cryptographic hardness.

### ML Attack on LWE
```
  Dimension n |  ML Accuracy | Random Guess |               Status
  ----------- | ------------ | ------------ | --------------------
            2 |        85.3% |        50.0% |   ⚠️  ML can break!
            8 |        53.1% |        50.0% |     ✅ LWE is SECURE
           16 |        49.8% |        50.0% |     ✅ LWE is SECURE
           32 |        50.2% |        50.0% |     ✅ LWE is SECURE
```

### Encrypted ML Inference
```
     Patient |             Raw Data | Plaintext Score | Encrypted Score | Match?
  Patient A  |    [55, 130, 210]    |             110 |             112 |     ✅
  Patient B  |    [25, 110, 170]    |            -120 |            -119 |     ✅
  Patient C  |    [65, 160, 250]    |             360 |             361 |     ✅
```

---

## 🔮 Future Scope

1. **Global PQC Migration (NIST 2035)** — Apple, Google, Microsoft are already transitioning to ML-KEM/ML-DSA
2. **Hybrid Cryptographic Systems** — combining classical (ECDH) + post-quantum (ML-KEM) for defense-in-depth
3. **Advanced Lattice Primitives** — Fully Homomorphic Encryption (FHE), post-quantum zero-knowledge proofs
4. **ML + PQC** — ML-assisted cryptanalysis (BKZ improvement), privacy-preserving ML (FHE + inference), ML-based side-channel attacks and defenses

---

## 📚 References

- **FIPS 203** — ML-KEM: Module-Lattice-Based Key-Encapsulation Mechanism Standard (NIST, 2024)
- **FIPS 204** — ML-DSA: Module-Lattice-Based Digital Signature Algorithm (NIST, 2024)
- **Regev, O. (2010)** — *"The Learning with Errors Problem"*, IEEE CCC
- **Pérez-Ramos et al. (2026)** — *"An interactive tool for teaching lattice-based post-quantum cryptography"*
- **Wenger et al. (2022)** — *"SALSA: Attacking Lattice Cryptography with Transformers"*
- **Gilad-Bachrach et al. (2016)** — *"CryptoNets: Applying Neural Networks to Encrypted Data"*

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open an **issue** for bugs or feature requests
- Submit a **pull request** with improvements
- Add new ML experiments or visualization enhancements

---

## ✍️ Author

**Jashan Goyal**

---

*Built with ❤️ for understanding the future of cryptography.*
