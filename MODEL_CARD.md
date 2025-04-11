# 📄 MODEL CARD for SCOPE-DTI

**Model Name:** SCOPE-DTI  
**Version:** v1.0  
**Paper:** [arXiv:2503.09251](https://arxiv.org/abs/2503.09251)  
**Authors:** Yigang Chen, Fezzy Zhang, et al.  
**Institution:** The Chinese University of Hong Kong, Shenzhen  
**License:** MIT License  
**Contact:** [GitHub Issues](https://github.com/Yigang-Chen/SCOPE-DTI/issues)  

---

## Model Overview

**SCOPE-DTI** (Semi-Inductive Dataset Construction and Framework Optimization for Practical Usability Enhancement in Deep Learning-Based Drug Target Interaction Prediction) is a unified deep learning framework for DTI prediction.

It combines:
- A **large-scale, semi-inductive DTI dataset**
- A GNN-based encoder for 3D chemical and protein information

---

## Intended Use

### Primary Use Cases
- Drug-target interaction (DTI) prediction for human proteins.
- Evaluating generalizability in semi-inductive settings (unseen protein-ligand pairs).
- Assisting bioinformatics research with efficient inference workflows.

### Target Users
- Computational biology researchers
- AI drug discovery engineers
- Developers building ML-based bioinformatics tools

### Out-of-Scope Uses
- **Clinical applications** without further validation
- **Drug efficacy/toxicity predictions**
- Any real-world use that affects **human health decisions**

---

## Model Architecture

- **Backbone:** Multi-module GNN integrating protein residue graphs and molecular graphs
- **Protein Representation:** Spatial graph using residue coordinates from AlphaFold
- **Compound Representation:** RDKit + DGLLifeTools molecular graphs
- **Framework:** PyTorch + DGL + PyG
- **Configuration:** YACS-based hierarchical configuration management

---

## Datasets

**Primary Dataset:** [SCOPE-DTI Total](https://awi.cuhk.edu.cn/SCOPE/downloads)  
- Contains: Protein-ligand pairs with binding interaction labels
- Includes 3D protein structures from AlphaFold
- Python script for semi-inductive DTI train/val/test dataset construction

**Format:**
- Protein 3D repersentation: `.pkl` with `"uniprot_id"` and `"crod"` (shape = `(L, 3)`)
- Dataset: `.parquet` with compound SMILES, SDF structure, ChemBL ID, protein uniprot ID and sequence

---

## Evaluation

### Metrics Used
- ROC-AUC
- PR-AUC
- F1-score
- Accuracy

### Hardware
- NVIDIA 2080Ti GPU (~25 min runtime on demo, 4GB VRAM)
- Our model is trained on NVIDIA A100 GPU (SCOPE-DTI-total, ~50h runtime, 40GB VRAM)

### Results (Demo)
On a sample of 10 proteins with semi-inductive split:
- ROC-AUC: ~0.84 (detailed metrics in `RESULT.OUTPUT_DIR/metrics.json`)

---

## ⚖️ Ethical Considerations

### Dataset Bias
- Protein coverage limited to AlphaFold-available structures
- Ligand diversity dependent on public repositories
- Potential bias toward well-studied proteins

### Limitations
- Generalization beyond the training set (e.g., novel scaffolds) may be limited
- Not suitable for structure-less proteins
- Interpretability is not the focus (although compatible with post hoc tools)

### Privacy & Security
- No personal or sensitive data involved
- All data used is from public databases

---

## Deployment & Tools

- **Demo:** `SCOPE-DTI-demo.ipynb`
- **Docker:** [scope_web DockerHub](https://hub.docker.com/r/zcorn/scope_web)
- **Lightweight Inference Repo:** [Lightweight-SCOPE-DTI-for-Inference](https://github.com/Yigang-Chen/Lightweight-SCOPE-DTI-for-Inference)

---

## Citation

If you use this work, please cite:

```bibtex
@article{chen2024scopedti,
  title={SCOPE-DTI: Semi-Inductive Dataset Construction and Framework Optimization for Practical Usability Enhancement in Deep Learning-Based Drug Target Interaction Prediction},
  author={Chen, Yigang and Zhang, Fezzy and others},
  journal={arXiv preprint arXiv:2503.09251},
  year={2024}
}
```

---

## Further Notes

* This project is part of a broader effort to make DTI prediction more **practically usable** in semi-inductive settings.
* Contributions and feedback are welcome via GitHub Issues.
