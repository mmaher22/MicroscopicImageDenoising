## Fluroscent Microscopic Image Denoising Using Deep Learning

In this project, we have used five different neural network architectures to remove noise from fluroscent microscopic images that were exposed to light for a shorter time period than ground truth images.

<hr>

### Architectures:
1. CNN with MSE Loss
2. CNN with VGG Loss
3. WGAN + MSE Loss for the generator
4. WGAN + VGG Loss for the generator
5. ResUNet GAN

All Architectures Implementations can be found at ``` ./Network Architectures ```

<hr>

### Noise Modeling:
To model the noise addition to input images, poisson noise is used and the functions used for noise modeling in addition to generation of some synthetic sine wave images can be found in ```Noise&Sines_Generator.ipynb```

<hr>

### Analysis and Evaluation:
```Analysis&Evaluation_Notebook.ipynb``` is the notebook used for evaluation and analysis of the results, comparing output with baseline denoising filters (Mean/Median/Gaussian/TV), computation of peak signal to noise ratio, computation structural similarity, and evaluation on sine wave images with different base levels, amplitudes, and wavelengths as a further analysis.

<hr>

### Sample Output:

##### Baseline Filters:
<div style="text-align: center">
<img src="https://cdn-images-1.medium.com/max/2400/1*QSedlO9j_1h24bLuwSVV_g.jpeg" width="900px" alt="Baseline Filters"/>
</div>

##### Neural Networks
<div style="text-align: center">
<img src = "https://cdn-images-1.medium.com/max/2400/1*uN9K233bahkih5C9alaJIg.jpeg" width="900px" alt = "Implemented Architectures"/>
</div>
<hr>

### Requirements:
- Python Tensorflow for CNN-MSE/CNN-VGG/WGAN-MSE/WGAN/VGG Architectures
- Python PyTorch for ResUNet Architecture

<br>

### References:
These references were followed to implement similar architectures
```
1. J. M. Wolterink, T. Leiner, M. A. Viergever, and I. Isgum, “Generative adversarial networks for noise reduction in low-dose CT”, IEEE Trans.Med. Imag., Dec. 2017.
2. Yang, Qingsong, et al. “Low-dose CT image denoising using a generative adversarial network with Wasserstein distance and perceptual loss.” IEEE transactions on medical imaging 37.6 (2018): 1348–1357.
3. Nie, Dong, et al. “Medical image synthesis with deep convolutional adversarial networks.” IEEE Transactions on Biomedical Engineering 65.12 (2018): 2720–2730.
```
