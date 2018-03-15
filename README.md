# Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering

The code in this repository implements an efficient generalization of the
popular Convolutional Neural Networks (CNNs) to arbitrary graphs, presented in
our paper:

Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst, [Convolutional Neural
Networks on Graphs with Fast Localized Spectral Filtering][arXiv], Neural
Information Processing Systems (NIPS), 2016.

Additional material:
* [NIPS2016 spotlight video][video], 2016-11-22.
* [Deep Learning on Graphs][slides_ntds], a lecture for EPFL's master course [A
  Network Tour of Data Science][ntds], 2016-12-21.
* [Deep Learning on Graphs][slides_dlid], an invited talk at the [Deep Learning on
  Irregular Domains][dlid] workshop of BMVC, 2017-09-17.

[video]: https://www.youtube.com/watch?v=cIA_m7vwOVQ
[slides_ntds]: https://doi.org/10.6084/m9.figshare.4491686
[ntds]: https://github.com/mdeff/ntds_2016
[slides_dlid]: https://doi.org/10.6084/m9.figshare.5394805
[dlid]: http://dlid.swansea.ac.uk

There is also implementations of the filters used in:
* Joan Bruna, Wojciech Zaremba, Arthur Szlam, Yann LeCun, [Spectral Networks
  and Locally Connected Networks on Graphs][bruna], International Conference on
  Learning Representations (ICLR), 2014.
* Mikael Henaff, Joan Bruna and Yann LeCun, [Deep Convolutional Networks on
  Graph-Structured Data][henaff], arXiv, 2015.

[arXiv]:  https://arxiv.org/abs/1606.09375
[bruna]:  https://arxiv.org/abs/1312.6203
[henaff]: https://arxiv.org/abs/1506.05163

## Installation

1. Clone this repository.
   ```sh
   git clone https://github.com/mdeff/cnn_graph
   cd cnn_graph
   ```

2. Install the dependencies. The code should run with TensorFlow 1.0 and newer.
   ```sh
   pip install -r requirements.txt  # or make install
   ```

3. Play with the Jupyter notebooks.
   ```sh
   jupyter notebook
   ```

## Reproducing our results

Run all the notebooks to reproduce the experiments on
[MNIST](nips2016/mnist.ipynb) and [20NEWS](nips2016/20news.ipynb) presented in
the paper.
```sh
cd nips2016
make
```

## Using the model

To use our graph ConvNet on your data, you need:

1. a data matrix where each row is a sample and each column is a feature,
2. a target vector,
3. optionally, an adjacency matrix which encodes the structure as a graph.

See the [usage notebook][usage] for a simple example with fabricated data.
Please get in touch if you are unsure about applying the model to a different
setting.

[usage]: http://nbviewer.jupyter.org/github/mdeff/cnn_graph/blob/outputs/usage.ipynb

## License & co

The code in this repository is released under the terms of the [MIT license](LICENSE.txt).
Please cite our [paper][arXiv] if you use it.

```
@inproceedings{cnn_graph,
  title = {Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering},
  author = {Defferrard, Micha\"el and Bresson, Xavier and Vandergheynst, Pierre},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2016},
  url = {https://arxiv.org/abs/1606.09375},
}
```
