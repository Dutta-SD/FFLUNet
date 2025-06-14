# FFLUNet: Feature Fused Lightweight UNET for Brain Tumor Segmentation

[![Paper](https://img.shields.io/badge/Paper-Computers%20in%20Biology%20and%20Medicine-blue.svg)](https://doi.org/XX.XXXX/XXX)  
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

---

**FFLUNet** is a novel, lightweight U-Net variant that incorporates multi-view feature fusion modules to enhance segmentation accuracy while reducing computational overhead. This model is designed for efficient and effective medical image segmentation and has been evaluated on multiple benchmark datasets.

## ✨ Highlights

- 🔬 Designed for 3D medical image segmentation
- 🧠 Enhanced feature fusion between encoder-decoder paths
- 🚀 Lightweight architecture with competitive performance
- 📉 Fewer parameters (1.45M) and faster inference than standard U-Net and nnU-Net

---

## 📄 Paper

> **Title**: FFLUNet: Feature Fused Lightweight UNET for Brain Tumor Segmentation  
> **Authors**: Surajit Kundu, Sandip Dutta, Jayanta Mukhopadhyay, Nishant Chakravorty
> **Journal**: Computers in Biology and Medicine  
> **DOI**: [https://doi.org/10.1016/j.compbiomed.2025.110460](https://doi.org/10.1016/j.compbiomed.2025.110460)

---

## 🧠 Model Architecture

The FFLUNet architecture enhances the classical U-Net by integrating feature fusion blocks that aggregate spatial and semantic information across layers. This fusion is both **progressive and multi-scale**, leading to better context understanding.

<p align="center">
  <img src="[assets/fflunet_architecture.png](https://ars.els-cdn.com/content/image/1-s2.0-S001048252500811X-gr1_lrg.jpg)" width="700"/>
</p>

---

## 📁 Datasets

FFLUNet has been tested on the following datasets:

- 🧠 **BraTS 2020** – Brain tumor MRI segmentation  
- 🧠 **BraTS Africa Glioma** – Brain tumor MRI segmentation 
- 🫀 **ACDC** – Cardiac MRI segmentation  

> **Note**: Due to license restrictions, the datasets are not included. You can download them from their respective official repositories.

---

## ⚙️ Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/your-username/FFLUNet.git
cd FFLUNet
pip install -e .

```
#### Experiment planning and preprocessing
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
#### 3D full resolution U-Net training
```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainer_FFLUNet
```

### 📚 More Information

For detailed documentation, setup instructions, and usage guidelines, please refer to the official nnU-Net repository:

🔗 [https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation](https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation)


