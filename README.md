# NMF_CUDA
Implementation of NMF algorithm in CUDA via python

### About

This repository implements a number of Non-negative matrix facotrization (NMF) algorithms for Nvidia CUDA-GPUs using the python libraries (**a**) [cudamat](https://github.com/cudamat/cudamat) and (**b**) [PyCUDA](https://mathema.tician.de/software/pycuda/) and [scikit-cuda](https://github.com/lebedov/scikit-cuda). The project was designed as a replacement for [NMF-GPU](https://github.com/bioinfo-cnb/bionmf-gpu) by [*Mejia-Roa et al.*](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0485-4) that only implements Kullbach-Leibler divergence minimizing NMF. We use it to run different NMF algorithms on the GPU in [the bratwurst project](https://github.com/wurst-theke/bratwurst).

### Dependencies

The project is designed so that only some of the used libries need to be installed. You can either run the solver on any normal CPU via numpy, or on CUDA capable GPUs via cudamat or PyCUDA and skcuda.

### Solver

NMF-CUDA includes the following NMF solver:

- original NMF by *Lee & Seung* [1999](http://www.nature.com/nature/journal/v401/n6755/full/401788a0.html), [2001](http://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization) for both cost functions Euclidean norm and Kullbach-Leibler divergence
- sparse NMF (i.e spare results, not deconvolution of sparse matrices)
- affine NMF from *Laurberg & Hansen* [2007](http://ieeexplore.ieee.org/document/4217493/)
- semiNMF and convexNMF from *Ding et al.* [2010](https://www.ncbi.nlm.nih.gov/pubmed/19926898)

### Usage

Running `nmf.py` gives the command line arguments. The project basically takes one large matrix (either a tabseparated text file or numpy matrix) and returns two (three for affine NMF) files of the same format for each NMF run.

### Project Structure

The file `nmf.py` is the command line callable script of the whole project. `NMFbase.py` provides the NMF base object(s) that are then customized for the three used libraries (library combinations).
