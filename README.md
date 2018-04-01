# Surface Networks

[Ilya Kostrikov](https://scholar.google.com/citations?user=PTS2AOgAAAAJ&hl=en), 
[Zhongshi Jiang](https://cs.nyu.edu/~zhongshi), 
[Daniele Panozzo](https://cs.nyu.edu/~panozzo), 
[Denis Zorin](http://mrl.nyu.edu/~dzorin/), 
[Joan Bruna](https://cims.nyu.edu/~bruna/)

*IEEE Conference on Computer Vision and Pattern Recognition* **CVPR 2018** 
**(Oral)**

## Abstract
We study data-driven representations 
for three-dimensional triangle meshes, which are one of the prevalent objects used to represent 3D geometry. 
Recent works have developed models
that exploit the intrinsic geometry of manifolds and graphs, 
namely the Graph Neural Networks (GNNs) and its spectral variants, 
which learn from the local metric tensor via the Laplacian operator. 

Despite offering excellent sample complexity and built-in invariances,
intrinsic geometry alone is invariant to isometric deformations, making it unsuitable for  many applications.
To overcome this limitation,
we propose several upgrades to GNNs 
to leverage extrinsic differential geometry properties 
 of three-dimensional surfaces, increasing its modeling power. 
 In particular, we propose to exploit the Dirac operator, whose spectrum detects principal curvature directions --- this is in stark contrast with the classical Laplace 
 operator, which directly measures mean curvature. We coin the 
 resulting models *Surface Networks (SN)*.

We prove that these models define shape representations that are stable to deformation and to discretization, and we demonstrate the efficiency and versatility of SNs on 
 two
 challenging tasks: temporal prediction of mesh deformations
 under non-linear dynamics and generative models using 
 a variational autoencoder framework with encoders/decoders
 given by SNs.

 ## Source Code
 Source code is hosted on this GitHub repository. Instructions can be read from the argparse options. 

 ## Requirements
 
pytorch, plyfile, numpy, scipy, progressbar, cupy, pynvrtc

If you are using Python 3, please install pynvrtc [here](https://github.com/jiangzhongshi/pynvrtc).

[Python bindings for libigl](https://github.com/libigl/libigl/tree/master/python) is used for geometry processing

## License
Source code [MPL2](http://www.mozilla.org/MPL/2.0/) licensed
([FAQ](http://www.mozilla.org/MPL/2.0/FAQ.html)). 

Please cite our paper if it helps.

```
@inproceedings{kostrikov2018surface,
  title={Surface Networks},
  author={Kostrikov, Ilya and Jiang, Zhongshi and Panozzo, Daniele and Zorin, Denis and Burna Joan},
  booktitle={2018 {IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR} 2018},
  year={2018}
}
```