# Machine Learning for Image Denoising

This project focuses on implementing and experimenting with **FFDNet** algorithm, as described in the paper ["FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising"](https://arxiv.org/abs/1710.04026). Our goal is to explore its performance on various image datasets and evaluate its robustness when trained and tested on different types of noise.
We will also explore the use of **diffusion models** for image denoising, aiming to understand and compare their methodology with CNN-based approaches like FFDNet.

---

## Authors

- **Rémi Calvet** : [@RemiCSK](https://github.com/RemiCSK)
- **Romain Delhommais** : [@Luuumine](https://github.com/Luuumine)
- **Nathan Van Assche** : [@NathanVnh](https://github.com/NathanVnh)

---

## Objectives

1. **Implement FFDNet**: Implement the FFDNet algorithm for image denoising, with a focus on performance across different types of noise and datasets.
2. **Evaluate performance**: Test FFDNet on various image datasets such as *ImageNet*, exploring various types of noise such as Gaussian noise and Box blur.
3. **Explore diffusion models**: Investigate the use of diffusion models for image denoising and compare their results with CNN-bases methods.
4. **Analysis**: Analyse the performance of both techniques, highlighting strengths, weaknesses, and areas for improvement.

---

## Installation

To set up and run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/RemiCSK/CNN-based-Image-Denoising
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Acknowledgments
- The authors of **FFDNet** for their original researchand implementation.
- Researchers contributing to the development of diffusion models in image denoising
- **ImageNet** for providing a large-scale, high-quality image dataset.