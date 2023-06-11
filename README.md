<h1 align="center">GigaTorch</h1>
<div align="center">
  <img alt="GigaTorch Logo" src="https://github.com/orifmilod/GigaTorch/assets/25881325/f9c41ced-f01a-4511-a301-6c18e64b02d3" width=235/>
  <br/>
  GigaTorch: Something between <a href="https://github.com/pytorch/pytorch">PyTorch</a> and <a href="https://github.com/geohot/tinygrad">TinyGrad</a>. Maintained by GigaChads around the world.
</div>
<br/>  <br/>


Welcome to GigaTorch, the revolutionary machine learning framework that takes the art of neural network training to unprecedented levels. It is not just any machine learning framework, GigaTorch is a true game-changer that caters specifically to the needs of GigaChads, those exceptional individuals in the field of deep learning.

GigaTorch finds the perfect balance between power and simplicity, rivaling the likes of PyTorch and Tinygrad. It offers an unparalleled ML ecosystem designed to unlock the full potential of GigaChad neural networks, enabling practitioners to achieve remarkable results with ease.

Join the ranks of GigaChads and harness the power of GigaTorch to propel your machine learning projects to unparalleled heights. Together, let's shape the future of artificial intelligence as GigaChads.

![Screenshot 2023-06-11 at 13 41 13](https://github.com/orifmilod/GigaTorch/assets/25881325/ad2f64c8-d8b1-4d45-a3e0-c7e77785edea) 
<p align='center'> Simple Feed Forward Neural Net with Giga Neurons for classifying GigaChads. <p/>
<br/> 

Key Features: (it's actually our vision)

1. GigaChad Exclusive: GigaTorch is purpose-built to harness the full potential of the GigaChad neural network architecture. By providing a dedicated ecosystem for GigaChad-based models, GigaTorch maximizes their efficiency and performance.

2. Unparalleled Performance: GigaTorch takes advantage of cutting-edge optimizations and novel algorithms, enabling lightning-fast model training and inference. With its streamlined execution pipeline, GigaTorch minimizes computational bottlenecks, resulting in significantly reduced training times.

3. Advanced Model Customization: GigaTorch empowers users to effortlessly design and tailor GigaChad networks to their specific needs. Leverage an extensive library of pre-defined layers, activation functions, and optimization algorithms.

4. Scalability and Parallelism: GigaTorch seamlessly scales across multiple GPUs and distributed systems, enabling efficient training of large-scale GigaChad models. With built-in support for data parallelism and model parallelism, GigaTorch empowers practitioners to tackle even the most demanding machine learning challenges.

Embrace the power of GigaChad and experience the next generation of machine learning with GigaTorch. Join the GigaTorch community and unlock the true potential of your GigaChad networks. The future of AI begins here.

## Installation

The current recommended way to install GigaTorch is from source.

#### From source
```sh
git clone https://github.com/orifmilod/GigaTorch.git
cd GigaTorch
python3 -m pip install -e . # or `py3 -m pip install -e .` if you are on windows
```

#### Building the project
```
python setup.py build
``` 

#### Run the tests 
```
python3 -m pytest 
```

#### Linting and formating the code in-place
```
python setup.py format
``` 
 
## Contributing & features to add
If you want to contribute, please take a look at the issues created and project tab, but here is list of features we want to add

- [x] Add AutoGrad https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html
- [x] Create Convolution neural-network using GigaTorch.
- [ ] Create Recurrent neural-network using GigaTorch. 
- [ ] Add accelerator support in Tensor class. 
 
Supported accelerators:
- [x] CPU
- [ ] CUDA (WIP)
- [ ] GPU (OpenCL)
- [ ] ANE (Apple Neural Engine)
- [ ] Google TPU 
