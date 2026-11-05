[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21934704-blue)](https://doi.org/10.5281/zenodo.21934705)

# Adaptive Rank-Cluster Image Filter (ARC-Filter)

In this fork, I added a Fashionpedia benchmark to test how preprocessing affects RF-DETR segmentation. With mixed synthetic noise, ARC scored **30.82 mask AP** versus **27.96** without filtering. On clean images, ARC lowered mask AP by **0.31 points**. [Results and evaluation setup](#downstream-evaluation)

This repository contains the official high-performance implementation of the **Adaptive Rank-Cluster Image Filter (ARC-Filter)**, synchronized across two major scientific computing platforms: **MATLAB** and **Python (NumPy/SciPy)**.

The filter is specifically optimized for advanced Computer Vision tasks, medical imaging (e.g., high bit-depth X-Ray/CT scans), and scientific data analysis.

## Key Features
- **Cross-Platform Bit-Depth Integrity:** Flawless handling of 8-bit and 16-bit grayscale and RGB images (including complex LibTIFF biomedical data layers).
- **Speed Optimization:** Python implementation leverages fast vectorization via native NumPy array operations and memory-efficient matrix layout matching.
- **Color Space Preservation:** Advanced RGB ↔ HSV transformation ensures noise filtering operates strictly on the luminance (Value) channel without altering chrominance metadata.

## Repository Structure
- `Images/` — Directory containing test image datasets (supports `.bmp`, `.jpg`, `.png`, `.tiff`, `.tif`).
- `main.py` — The optimized Python execution pipeline (automated folder scan, image processing, and strict MATLAB-compatible quantization).
- `arc_filter.py` — The core vectorized ARC-filter algorithm.
- `ARC_filtering.m` — Monolithic, high-speed MATLAB program containing both the execution pipeline and the internal ARC-filter algorithm.
- `scripts/` — Fashionpedia evaluation runner comparing no filter, median, bilateral, and ARC for a fixed RF-DETR Seg Small model.
- `requirements.txt` — Clean Python package dependencies list.
- `requirements-fashionpedia.txt` — Extra dependencies for the Fashionpedia evaluation (`scripts/arc_full_valid.py`).
- `LICENSE` — Official MIT License terms.

## Verification & Precision
Rigorous mathematical validation confirms **99.99% pixel-to-pixel structural identity** between the MATLAB and Python runtimes (verified via structural similarity metrics). 

*Note on cross-platform math:* Minor standalone differences ($\pm1$ intensity grade in edge-case contrast regions or color cascade spaces) represent standard platform-specific floating-point arithmetic variances (Intel MKL vs SciPy/Cephes `erf` approximations and compiler-specific register precisions during high-contrast threshold division).

## Downstream Evaluation

We evaluated ARC as a preprocessing step for a fixed RF-DETR Seg Small model on 1,158 Fashionpedia validation images. Under mixed Gaussian noise (σ = 0.10 on a [0, 1] scale) and 1% salt-and-pepper noise, ARC achieved 30.82 mask AP, compared with 29.27 for median filtering, 27.96 without filtering, and 27.87 for bilateral filtering. Results were averaged over three noise seeds at a detection score cutoff of 0.05. On the original images, ARC reduced mask AP by 0.31 points relative to no filtering. In a separate 30-image check with one noise seed, lowering the cutoff from 0.05 to 0.001 narrowed ARC’s lead over median filtering from 1.26 to 0.59 points. Install the evaluation dependencies with `pip install -r requirements-fashionpedia.txt`. The experiment requires the Fashionpedia validation split and the Fashionpedia-finetuned RF-DETR Seg Small checkpoint used here. The weights are not included in this repository. Run `python scripts/arc_full_valid.py --help` for the expected paths and evaluation options.

## Quick Start (Python)
1. Clone the repository and navigate to the project root.
2. Place your source images into the `Images/` subfolder.
3. Install the clean required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the consecutive processing pipeline:
   ```bash
   python main.py
   ```
Filtered images will be automatically saved into the same `Images/` directory with the `_out(m)` suffix.

## AI Disclosure
The “Gemini 3.6 Flash” multimodal AI model by Google was utilized during the development phase of the software. Specifically, AI technologies were applied to solve the following tasks:
- Developing a software MATLAB script to generate a synthetic test image.
- Developing executable software scripts implementing the ARC filter algorithm.
- Performing deep optimization of the ARC filter code to reduce image processing time.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## How to Cite
If you apply these specific software implementations, please cite the GitHub repository:

```bibtex
@misc{ARC_Filter_2026,
  author       = {Pozdnyakov, Dmitry},
  title        = {{Adaptive Rank-Cluster Image Filter (ARC-Filter)}},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/DrDmitryPozdnyakov/ARC_Filter}},
  note         = {GitHub Repository}
}
```

If you use the developed algorithms, code, or findings in your scientific research, please cite the official arXiv paper:

```bibtex
@article{pozdnyakov2026image,
  author    = {Pozdnyakov, Dmitry},
  title     = {{Image Denoising via the Adaptive Rank-Cluster Filter}},
  journal   = {arXiv preprint arXiv:2608.15298},
  year      = {2026}
}
```
