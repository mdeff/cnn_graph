# Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering

The code in this repository implements an efficient generalization of the
popular Convolutional Neural Networks (CNNs) to arbitrary graphs, presented in
our paper:

Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst, [Convolutional Neural
Networks on Graphs with Fast Localized Spectral Filtering][arXiv], Neural
Information Processing Systems (NIPS), 2016.

The code is released under the terms of the [MIT license](LICENSE.txt). Please
cite the above paper if you use it.

Additional material:
* [NIPS2016 spotlight video][video], 2016-11-22.
* [NIPS2016 poster][poster]
* [Deep Learning on Graphs][slides_ntds], a lecture for EPFL's master course [A
  Network Tour of Data Science][ntds], 2016-12-21.
* [Deep Learning on Graphs][slides_dlid], an invited talk at the [Deep Learning on
  Irregular Domains][dlid] workshop of BMVC, 2017-09-17.
* most general
* Specific to the algorithm: Presentation at the Swiss Machine Learning Day
* More previous work: candidacy exam
* That [blog post] is a gentle introduction of the model.

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

## Experiments

* MNIST (NIPS2016)
* 20NEWS (NIPS2016)
* RCV1
* Wikipedia (NIPS2017)

Moving MNIST and PTB experiments were not conducted by me.

## Using the model

To use our graph ConvNet on your data, you need:

1. a data matrix where each row is a sample and each column is a feature,
2. a target vector,
3. optionally, an adjacency matrix which encodes the structure as a graph.

See the [usage notebook][usage] for a simple example with fabricated data.
Please get in touch if you are unsure about applying the model to a different
setting.

[usage]: http://nbviewer.jupyter.org/github/mdeff/cnn_graph/blob/outputs/usage.ipynb

## Applications

* [Kipf & Weiling '16] applied a first-order approximation of that model to
  a supervised learning task. A [blog post] by the author shows an interesting
  connection to the ll algorithm. A [blog post] by Ferenz provides a critical
  analysis of the method.

[kipf_paper]:
[kipf_blog]:

## Repository organization

See https://github.com/drivendata/cookiecutter-data-science/tree/master/%7B%7B%20cookiecutter.repo_name%20%7D%7D

* The models (the introduced model and some reference models) are contained in [models.py](models.py).
* Various side functions are implemented in [graph.py](graph.py), [coarsening.py](coarsening.py) and [utils.py](utils.py).
* We did experiments on three datasets: MNIST ([notebook](mnist.ipynb)), 20NEWS ([notebook](20news.ipynb)) and RCV1 ([notebook](rcv1.ipynb)).
* TensorBoard summaries are saved in the `summaries` folder.
* Model parameters are saved in the `checkpoints` folder.
* Data is placed in the `data` folder.
	* [MNIST](http://yann.lecun.com/exdb/mnist/) is downloaded automatically.
	* [20NEWS](http://qwone.com/~jason/20Newsgroups/) (`20news-bydate.tar.gz`) is downloaded automatically.
	* [RCV1](http://trec.nist.gov/data/reuters/reuters.html) should be downloaded manually and placed in TODO.
	* [pre-trained word2vec embeddings](https://code.google.com/archive/p/word2vec/) (`GoogleNews-vectors-negative300.bin.gz`).
	* Wikipedia graph and activations are available here. Please cite .. if you use it.
* The [trials](trials) folder contains various small experiences in the form of IPython notebooks.
	1. [Learning graph filters][trial1]: first experiments on learning
	   synthesized graph filters through observations of filtered and source
	   graph signals. The Chebychev and Lanczos methods as well as optimization
	   methods are compared there.
	2. [Classification][trial2]: learning filters who extract good features for
	   classification.
	3. [TensorFlow][trial3]: first experience with TensorFlow.
	4. [Coarsening][trial4]: implementation of the Graclus coarsening algorithm
	   and comparison with a previous matlab implementation.
* A [makefile](makefile) who runs every notebook as a sanity check. It only runs the code, there is no check on the results.

[trial1]: h

## Contributing

* Please fill a GitHub issue if you encounter any problem. Issues are better than contacting the authors as the community can respond and
* Pull requests are welcome !
* You can contact me for any help regarding how to apply our model to your problem.
