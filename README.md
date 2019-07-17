# POLY-GAN
Poly-GAN: Multi-Conditioned GAN for Fashion Synthesis
# Abstract

We present Poly-GAN, a novel conditional GAN architecture that is motivated
by Fashion Synthesis, an application where garments are automatically placed on
images of human models at an arbitrary pose. Poly-GAN allows conditioning on
multiple inputs and is suitable for many tasks, including image alignment, image
stitching and inpainting. Existing methods have a similar pipeline where three
different networks are used to first align garments with the human pose, then
perform stitching of the aligned garment and finally refine the results. Poly-GAN
is the first instance where a common architecture is used to perform all three tasks.
Our novel architecture enforces the conditions at all layers of the encoder and
utilizes skip connections from the coarse layers of the encoder to the respective
layers of the decoder. Poly-GAN is able to perform a spatial transformation of the
garment based on the RGB skeleton of the model at an arbitrary pose. Additionally,
Poly-GAN can perform image stitching, regardless of the garment orientation,
and inpainting on the garment mask when it contains irregular holes. Our system
achieves state-of-the-art quantitative results on Structural Similarity Index metric
and Inception Score metric using the DeepFashion dataset.


# Architecture of POLY-GAN
