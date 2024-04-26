<h1 align="center">GigaTorch</h1>
<div align="center">
  <img alt="GigaTorch Logo" src="https://github.com/orifmilod/GigaTorch/assets/25881325/f9c41ced-f01a-4511-a301-6c18e64b02d3" width=235/>
  <br/>
  GigaTorch: Something between <a href="https://github.com/pytorch/pytorch">PyTorch</a> and <a href="https://github.com/geohot/tinygrad">TinyGrad</a>. Maintained by GigaChads around the world.
</div>
<br/>  <br/>


Welcome to GigaTorch, the revolutionary machine learning framework that takes the art of neural network training to unprecedented levels. It is not just any machine learning framework, GigaTorch is a true game-changer that caters specifically to the needs of GigaChads, those exceptional individuals in the field of deep learning.

![Screenshot 2023-06-11 at 13 41 13](https://github.com/orifmilod/GigaTorch/assets/25881325/ad2f64c8-d8b1-4d45-a3e0-c7e77785edea) 
<p align='center'> Simple Feed Forward Neural Net with Giga Neurons for classifying GigaChads. <p/>
<br/> 

Key Features: (it's our vision for now)

1. Unparalleled Performance: GigaTorch takes advantage of cutting-edge optimizations and novel algorithms, enabling lightning-fast model training and inference. With its streamlined execution pipeline, GigaTorch minimizes computational bottlenecks, resulting in significantly reduced training times.

2. Advanced Model Customization: GigaTorch empowers users to effortlessly design and tailor GigaChad networks to their specific needs. Leverage an extensive library of pre-defined layers, activation functions, and optimization algorithms.

3. Scalability and Parallelism: GigaTorch seamlessly scales across multiple GPUs and distributed systems, enabling efficient training of large-scale GigaChad models. With built-in support for data parallelism and model parallelism, GigaTorch empowers practitioners to tackle even the most demanding machine learning challenges.

Embrace the power of GigaChad and experience the next generation of machine learning with GigaTorch. Join the GigaTorch community and unlock the true potential of your GigaChad networks. The future of AI begins here.
## Current implemented examples

- [X] MLP based [Character Level Language Model](https://github.com/orifmilod/GigaTorch/blob/master/example/language_model/language-model.py) following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [X] [CNN](https://github.com/orifmilod/GigaTorch/blob/master/gigatorch/cnn.py) for classifying MNIST dataset.

## Installation

The current recommended way to install GigaTorch is from source.

#### From source
```sh
git clone https://github.com/orifmilod/GigaTorch.git
cd GigaTorch
# Make sure you have virtualenv installed
pip3 install poetry
poetry config virtualenvs.in-project true
poetry install
```
#### Building the project
```
poetry build
``` 

#### Run the tests 
```
poetry run pytest 
```

#### Linting and formating the code in-place
```
poetry run black .
``` 
#### Running examples
poetry run python example/image_models/gigatorch/cnn.py
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
