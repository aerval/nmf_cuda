# NMF algorithms in normal numpy without parallelization

import numpy as np
import NMFbase

class NMF(NMFbase.NMFbase):

    def run(self, iterations):
        """Run the nmf for given number of iterations or until converged"""

        for i in range(0, iterations):
            # update H
            self.H = self.H * np.dot(self.W.T,self.X) \
                              / np.dot(np.dot(self.W.T,self.W),self.H)

            # update W
            self.W = self.W * np.dot(self.X,self.H.T) \
                              / np.dot(np.dot(self.W, self.H), self.H.T)

            # test for convergence
            if (i % self.niter_test_conv == 0) and self.checkConvergence():
                print "NMF converged after %i iterations" % i
                break

class NMFsparse(NMFbase.NMFsparse, NMFbase.NMFbase):

    def run(self, iterations):

        for i in range(0, iterations):
            # A = W, X = H, Y = X, Yhat = WH
            WH = np.dot(self.W,self.H)
            self.W = self.W * np.dot(self.X, self.H.T) \
                              / (np.dot(WH, self.H.T) + self.sparseW)
            self.W = self.W / self.W.sum(0) #[:,None]
                   # alternative: Wn = (W.T / W.sum(1)).T
            self.H = self.H * np.dot(self.W.T, self.X) \
                              / (np.dot(self.W.T, WH) + self.sparseH)

            # test for convergence
            if (i % self.niter_test_conv == 0) and self.checkConvergence():
                print "NMF converged after %i iterations" % i
                break

class NMFaffine(NMFbase.NMFaffine, NMFbase.NMFbase):

    def run(self, iterations):

        Xsum = self.X.sum(1)

        for i in range(0, iterations):
            # A = W, X = H, Y = X, Yhat = WH
            WH = np.dot(self.W,self.H) + self.W0[:,None]
            self.W = self.W * np.dot(self.X, self.H.T) \
                              / (np.dot(WH, self.H.T) + self.sparseW)
            self.W = self.W / self.W.sum(0) #[:,None]
                   # alternative: Wn = (W.T / W.sum(1)).T
            self.H = self.H * np.dot(self.W.T, self.X) \
                              / (np.dot(self.W.T, WH) + self.sparseH)

            self.W0 = self.W0 * Xsum / WH.sum(1)

            # test for convergence
            if (i % self.niter_test_conv == 0) and self.checkConvergence():
                print "NMF converged after %i iterations" % i
                break

class NMFsemi(NMFbase.NMFsemi, NMFbase.NMFbase):

    def run(self, iterations):
        for i in range(iterations):
            try:
                self.F = np.dot(np.dot(self.X, self.G),
                                 np.linalg.inv(np.dot(self.G.T, self.G))
                                )
            except LinAlgError:
                self.F = np.dot(np.dot(self.X, self.G),
                                 np.linalg.pinv(np.dot(self.G.T, self.G))
                                )
            XTF = np.dot(self.X.T, self.F)
            FTF = np.dot(self.F.T, self.F)

            XTF_pos = (np.abs(XTF) + XTF) / 2
            XTF_neg = XTF_pos - XTF
            FTF_pos = (np.abs(FTF) + FTF) / 2
            FTF_neg = FTF_pos - FTF

            self.G = self.G * np.sqrt((XTF_pos + np.dot(self.G, FTF_neg) \
                                                 + 10**-9) \
                                        / (XTF_neg + np.dot(self.G, FTF_pos) \
                                                   + 10**-9)
                                       )

            # test for convergence
            if (i % self.niter_test_conv == 0) and self.checkConvergence():
                print "NMF converged after %i iterations" % i
                break

class NMFconvex(NMFbase.NMFconvex, NMFbase.NMFbase):

    def run(self, iterations):
        XTX = np.dot(self.X.T, self.X)
        XTXpos = (np.abs(XTX) + XTX) / 2
        XTXneg = XTXpos - XTX

        for i in range(0, iterations):
            # Update G
            XTXnegW = np.dot(XTXneg, self.W)
            XTXposW = np.dot(XTXpos, self.W)
            GWT = np.dot(self.G, self.W.T)
            self.G *= np.sqrt((XTXposW + np.dot(GWT, XTXnegW)) \
                               / (XTXnegW + np.dot(GWT, XTXposW) \
                                          + 10**-9)
                              )

            # Update W
            GTG = np.dot(self.G.T, self.G)
            self.W *= np.sqrt( (np.dot(XTXpos, self.G) \
                                 + np.dot(XTXnegW, GTG)) \
                                / (np.dot(XTXneg, self.G) \
                                   + np.dot(XTXposW, GTG) + 10**-9)
                              )

            # test for convergence
            if (i % self.niter_test_conv == 0) and self.checkConvergence():
                print "NMF converged after %i iterations" % i
                break

class NMFKL(NMFbase.NMFbase):

    def run(self, iterations):

        for i in range(0, iterations):
            self.H = self.H * np.dot(self.W.T,self.X \
                                                 / np.dot(self.W,self.H)) \
                              / np.sum(self.W,0)[:,None]
            self.W = self.W * np.dot(self.X \
                                       / np.dot(self.W,self.H),self.H.T) \
                              / np.sum(self.H,1)

            # test for convergence
            if (i % self.niter_test_conv == 0) and self.checkConvergence():
                print "NMF converged after %i iterations" % i
                break
