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

1. **Explore the FFDNet Algorithm**  
   Investigate the architecture, training process, and performance of the FFDNet algorithm for image denoising, focusing on its ability to handle varying noise levels using a noise-level map.

2. **Implement FFDNet**  
   Implement the FFDNet algorithm with modifications to adapt to smaller datasets like MNIST and CIFAR, emphasizing computational efficiency and training scalability.

3. **Evaluate on MNIST Dataset and CIFAR Dataset**  
   Test the implemented FFDNet model on the MNIST and CIFAR datasets, analyzing its ability to denoise images and preserve fine details under varying noise conditions.

4. **Analyze Results and Applications**  
   Analyze the denoising results, emphasizing practical applications like medical imaging and surveillance, and explore potential future improvements for FFDNet.

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
3. Start the python file
   ```bash
   python testing_ffdnet_inspired_cifar.py
   ```
---
All the algorithms were trained in the train_{} files that created a special file in the trained_model folder with its name, and the test/testing_{} files are files that can be runned. The one we wrote here is the application for the CIFAR10 images but it can be replaced with MNIST or other named files in the folder.


## Acknowledgments
- The authors of **FFDNet** for their original research and implementation.
- Researchers contributing to the development of diffusion models in image denoising
- **ImageNet** for providing a large-scale, high-quality image dataset.
