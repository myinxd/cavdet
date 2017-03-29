# cavdet
Due to active galaxy nuclei (AGN) mechanism at the center of the galaxies, electronics are ejected to blow and push the gas around. Bubbles or cavities are then generated, which can be detected at the X-ray band. 
Since AGN reveals quite lots of attracting physical phenomenons, detecting of them is significant. However, there exist many difficulties disturbing our works. For instance, the background and system noise in the X-ray images, which lead to low signal-to-noise ratio (SNR), should be eliminated. In addition, the high brightness (temperature) in the galaxy center usually leads to low contrast of the ROI compared to other regions. 

## Methods
In this repository, scripts of three main cavity detecion methods are provided. They are the two widely used beta-model fitting, and the unsharp masking (UM) based method. As well as our newly proposed granular convolutional neural networks (GCNN) models.
### Beta-model fitting
With regard to the beta-model fitting method, it fits the center of the galaxy by a two dimensional function with an elliptical plain view, and subtract the fitted pattern from the raw images.  After subtraction, the cavities are usually more salient on the residual images. 

### Unsharp masking (UM)
As for the UM methods, the image segmentation thinking is applied. They convolve the raw image to two Gaussian kernels with different variances, and subtract or divide the two convolved images, so as to improve the contrast of the target structures.

### Granular convolutional neural network (GCNN)
<TODO>

## Requirements
To process our scripts, some python packages are required, which are listed as follows.

- numpy, scipy, pickle 
- scikit-image
- [astropy](http://docs.astropy.org/en/stable/), [astroquery](http://astroquery.readthedocs.io/en/latest/)
- [Theano](http://www.deeplearning.net/software/theano/), [Lasagne](http://lasagne.readthedocs.io/en/latest/)

In addition, the compurtation can be accelerated by parallelly processing with GPUs. In this work, our scripts are written under the guide of Nvidia CUDA, thus the Nvidia GPU hardware is also required.

- CUDA Toolkit
  https://developer.nvidia.com/cuda-downloads


## References
- [Theano tutorial](http://www.deeplearning.net/software/theano/)
- [Lasagne tutorial](http://lasagne.readthedocs.io/en/latest/user/tutorial.html)
- [Save python data by pickle](http://www.cnblogs.com/pzxbc/archive/2012/03/18/2404715.html)
- [An image of diffuse emission](http://cxc.cfa.harvard.edu/ciao/threads/diffuse_emission/)
- [Angular diameter distance](https://en.wikipedia.org/wiki/Angular_diameter_distance)
- [astroquery.Ned](http://astroquery.readthedocs.io/en/latest/ned/ned.html)

## Author
- Zhixian MA <`zxma_sjtu(at)qq.com`>

## License
Unless otherwise declared:

- Codes developed are distributed under the [MIT license](https://opensource.org/licenses/mit-license.php);
- Documentations and products generated are distributed under the [Creative Commons Attribution 3.0 license](https://creativecommons.org/licenses/by/3.0/us/deed.en_US);
- Third-party codes and products used are distributed under their own licenses.

