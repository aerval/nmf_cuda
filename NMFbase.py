#!/usr/bin/python
# NMF base class from which all NMFs are derived

import numpy as np

class NMFbase(object):
    def __init__(self, X, rank, H=None, W=None):
        """Initializes the class"""
        self.X = X
        self.n, self.m = X.shape
        self.rank = rank
        self.H = H
        self.W = W

        self.setConvergenceMethod()

    def getH(self):
        return(self.H)

    def getW(self):
        return(self.W)

    def result(self):
        """Returns the generated matrices H and W"""
        return(self.getH(), self.getW())

    def setVariables(self):
        """allocates variables needed for running the NMF"""
        pass

    # run function

    def start(self, iterations):
        """Initializes and allocate all needed variables and then runs the
        algorithm"""
        self.initializeRandom()

        self.setVariables()

        self.run(iterations)

    """
                Convergence methods
    """

    def exposureChange(self):
        """Convergence method from bioNMF_GPU"""

        newExpo = np.argmax(self.getH(), axis=0)
        if (self.oldExposures != newExpo).any():
            self.oldExposures = newExpo
            self.const = 0
        else:
            self.const += 1
            if self.const == self.stop_threshold:
                return(True)
        return(False)

    def setConvergenceMethod(self, convergenceMethod="",
                             niter_test_conv = 1000, **kwargs):

        if convergenceMethod.lower() in ["bionmf_gpu", "exposurechange"]:
            self.checkConvergence = self.exposureChange
            self.oldExposures = np.zeros(self.m, dtype=np.int64)
            self.const = 0
            self.stop_threshold = kwargs['stop']

        else:
            self.checkConvergence = lambda: False

        self.niter_test_conv = niter_test_conv

    """
                Initialization methods
    """

    def initializeRandom(self, seed = 0, overwrite=False):
        """Initializes H and W randomly"""
        if seed != 0:
            np.random.seed(seed)

        if (self.H is None) or overwrite:
            self.H = np.random.random((self.rank,self.m)).astype(np.float32)

        if (self.W is None) or overwrite:
            self.W = np.random.random((self.n,self.rank)).astype(np.float32)

    """
                Distance methods
    """

    def frobError(self):
        return(np.linalg.norm(self.X-np.dot(self.getW(),self.getH()))
               / np.linalg.norm(self.X))

    def frobNorm(self):
        return(np.linalg.norm(self.X-np.dot(self.getW(),self.getH())))

    def getDistance(self, norm="frobError"):
        """Get a certain type of norm from string. Note that we do not specify
        what we return."""

        if norm.lower() in ["fn", "en", "frobnorm", "euclnorm",
                            "frobeniusnorm", "euclideannorm"]:
            return(self.FrobNorm())
        else:
            if not norm.lower() in ["fe", "ee", "froberror",
                                    "frobeniuserror"]:
                print """Unknown distance type %s. Returning the relative
                       frobenius error instead""" % norm

            return(self.FrobError())

"""
Objects for NMF variants with distinct methods
"""

class NMFsparse():

    def __init__(self, X, rank, H=None, W=None, sparseH=0., sparseW=0.):
        self.X = X
        self.n, self.m = X.shape
        self.rank = rank
        self.H = H
        self.W = W
        self.sparseH = sparseH
        self.sparseW = sparseW

        self.setConvergenceMethod()

class NMFaffine():

    def __init__(self, X, rank, H=None, W=None, sparseH=0., sparseW=0.):
        self.X = X
        self.n, self.m = X.shape
        self.rank = rank
        self.H = H
        self.W = W
        self.W0 = X.mean(1)
        self.sparseH = sparseH
        self.sparseW = sparseW

        self.setConvergenceMethod()

    def getW0(self):
        return(self.W0[:,None])

    def frobError(self):
        return(np.linalg.norm(self.X - np.dot(self.getW(),self.getH()) \
                                     - self.getW0()) \
               / np.linalg.norm(self.X))

    def frobNorm(self):
        return(np.linalg.norm(self.X - np.dot(self.getW(),self.getH()) \
                                     - self.getW0()))
class NMFsemi():

    def __init__(self, X, rank, G=None):
        self.X = X
        self.n, self.m = X.shape
        self.rank = rank
        self.G = G
        self.F = None

        self.setConvergenceMethod()

    def getH(self):
        return(self.G.T)

    def getW(self):
        return(self.F)

    def initializeRandom(self, seed = 0, overwrite=False):
        """Initializes G randomly"""
        if seed != 0:
            np.random.seed(seed)

        if (self.G is None) or overwrite:
            self.G = np.random.random((self.m,self.rank)).astype(np.float32)

class NMFconvex(NMFsemi):

    def __init__(self, X, rank, G=None):
        self.X = X
        self.n, self.m = X.shape
        self.rank = rank
        self.G = G
        self.W = None

        self.setConvergenceMethod()

    def getW(self):
        return(np.dot(self.X, self.W))

    def initializeRandom(self, seed = 0, overwrite=False):
        """Initializes G randomly"""
        if seed != 0:
            np.random.seed(seed)

        if (self.G is None) or overwrite:
            self.G = np.random.random((self.m,self.rank)).astype(np.float32)

        Wi = np.dot(self.G, np.linalg.inv(np.dot(self.G.T,self.G)))
        Wipos = (np.abs(Wi) + Wi) / 2
        self.W = Wipos + 0.2 * np.sum(np.abs(Wi)) / Wi.size
