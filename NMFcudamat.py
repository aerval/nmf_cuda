# NMF algorithms in cudamat

import numpy as np
import cudamat as cm
import NMFbase

# initialize the cudamat library
cm.cublas_init()

class NMFcudamat(NMFbase.NMFbase):

    def getH(self):
        return(self.H_gpu.asarray())

    def getW(self):
        return(self.W_gpu.asarray())

class NMF(NMFcudamat):

    def setVariables(self):

        self.H_gpu = cm.CUDAMatrix(self.H)
        self.W_gpu = cm.CUDAMatrix(self.W)
        self.X_gpu = cm.CUDAMatrix(self.X)
        self.WTW_gpu = cm.empty((self.rank, self.rank))
        self.WTWH_gpu = cm.empty(self.H.shape)
        self.WTX_gpu = cm.empty(self.H.shape)
        self.XHT_gpu = cm.empty(self.W.shape)
        self.WH_gpu = cm.empty(self.X.shape)
        self.WHHT_gpu = cm.empty(self.W.shape)

    def run(self, iterations):

        for i in range(0,iterations):
            # update H
            cm.dot(self.W_gpu.T, self.X_gpu, target=self.WTX_gpu)
            cm.dot(self.W_gpu.T, self.W_gpu, target=self.WTW_gpu)
            cm.dot(self.WTW_gpu, self.H_gpu, target=self.WTWH_gpu)
            self.H_gpu.mult(self.WTX_gpu).divide(self.WTWH_gpu)

            # update W
            cm.dot(self.X_gpu, self.H_gpu.T, target=self.XHT_gpu)
            cm.dot(self.W_gpu, self.H_gpu, target=self.WH_gpu)
            cm.dot(self.WH_gpu, self.H_gpu.T, target=self.WHHT_gpu)
            self.W_gpu.mult(self.XHT_gpu).divide(self.WHHT_gpu)

            # test for convergence
            if (i % self.niter_test_conv == 0) and self.checkConvergence():
                print "NMF converged after %i iterations" % i
                break

class NMFsparse(NMFbase.NMFsparse, NMFcudamat):

    def setVariables(self):
        self.H_gpu = cm.CUDAMatrix(self.H)
        self.W_gpu = cm.CUDAMatrix(self.W)
        self.X_gpu = cm.CUDAMatrix(self.X)
        self.XHT_gpu = cm.empty(self.W.shape)
        self.WH_gpu = cm.empty(self.X.shape)
        self.WHHT_gpu = cm.empty(self.W.shape)
        self.Wrowsum_gpu = cm.empty([1, self.rank])
        self.WTWH_gpu = cm.empty(self.H.shape)
        self.WTX_gpu = cm.empty(self.H.shape)

        ### helpers as sum is slower than cm.dot
        self.nones_gpu = cm.CUDAMatrix(np.ones([1,self.n]))

    def run(self, iterations):

        for i in range(0,iterations):
            # update W
            cm.dot(self.W_gpu, self.H_gpu, target=self.WH_gpu)
            cm.dot(self.X_gpu, self.H_gpu.T, target=self.XHT_gpu)
            cm.dot(self.WH_gpu, self.H_gpu.T, target=self.WHHT_gpu)
            self.WHHT_gpu.add(self.sparseW)
            self.W_gpu.mult(self.XHT_gpu).divide(self.WHHT_gpu)

            # normalize W
            cm.dot(self.nones_gpu, self.W_gpu, target=self.Wrowsum_gpu) # slower correct version: W_gpu.sum(0, target=rowsum_gpu)
            self.W_gpu.div_by_row(self.Wrowsum_gpu)

            # update H
            cm.dot(self.W_gpu.T, self.X_gpu, target=self.WTX_gpu)
            cm.dot(self.W_gpu.T, self.WH_gpu, target=self.WTWH_gpu)
            self.WTWH_gpu.add(self.sparseH)
            self.H_gpu.mult(self.WTX_gpu).divide(self.WTWH_gpu)

            # test for convergence
            if (i % self.niter_test_conv == 0) and self.checkConvergence():
                print "NMF converged after %i iterations" % i
                break

class NMFaffine(NMFbase.NMFaffine, NMFcudamat):

    def getW0(self):
        return(self.W0_gpu.asarray())

    def setVariables(self):
        self.H_gpu = cm.CUDAMatrix(self.H)
        self.W_gpu = cm.CUDAMatrix(self.W)
        self.X_gpu = cm.CUDAMatrix(self.X)
        self.W0_gpu = cm.CUDAMatrix(self.W0[:,None])
        self.XHT_gpu = cm.empty(self.W.shape)
        self.WH_gpu = cm.empty(self.X.shape)
        self.WHHT_gpu = cm.empty(self.W.shape)
        self.rowsum_gpu = cm.empty([1, self.rank])
        self.Xcolsum_gpu = cm.empty([self.n, 1])
        self.WHcolsum_gpu = cm.empty([self.n, 1])
        self.WTWH_gpu = cm.empty(self.H.shape)
        self.WTX_gpu = cm.empty(self.H.shape)

        ### helpers as sum is slower than cm.dot
        self.nones_gpu = cm.CUDAMatrix(np.ones([1,self.n]))
        self.mones_gpu = cm.CUDAMatrix(np.ones([self.m,1]))

        self.X_gpu.sum(1, target=self.Xcolsum_gpu)

    def run(self, iterations):

        for i in range(0,iterations):
            # update W
            cm.dot(self.W_gpu, self.H_gpu, target=self.WH_gpu)
            self.WH_gpu.add_col_vec(self.W0_gpu)
            cm.dot(self.X_gpu, self.H_gpu.T, target=self.XHT_gpu)
            cm.dot(self.WH_gpu, self.H_gpu.T, target=self.WHHT_gpu)
            self.WHHT_gpu.add(self.sparseW)
            self.W_gpu.mult(self.XHT_gpu).divide(self.WHHT_gpu)

            # normalize W
            cm.dot(self.nones_gpu, self.W_gpu, target=self.rowsum_gpu) # slower correct version: W_gpu.sum(0, target=rowsum_gpu)
            self.W_gpu.div_by_row(self.rowsum_gpu)

            # update H
            cm.dot(self.W_gpu.T, self.X_gpu, target=self.WTX_gpu)
            cm.dot(self.W_gpu.T, self.WH_gpu, target=self.WTWH_gpu)
            self.WTWH_gpu.add(self.sparseH)
            self.H_gpu.mult(self.WTX_gpu).divide(self.WTWH_gpu)

            # update W0
            cm.dot(self.WH_gpu, self.mones_gpu, target=self.WHcolsum_gpu) # slower correct version: WH_gpu.sum(1, target=WHcolsum_gpu)
            self.W0_gpu.mult(self.Xcolsum_gpu).divide(self.WHcolsum_gpu)

            # test for convergence
            if (i % self.niter_test_conv == 0) and self.checkConvergence():
                print "NMF converged after %i iterations" % i
                break

class NMFsemi(NMFbase.NMFsemi, NMFcudamat):

    def getH(self):
        return(self.G_gpu.asarray().T)

    def getW(self):
        return(self.F_gpu.asarray())

    def setVariables(self):
        n, m, r = self.n, self.m, self.rank

        self.G_gpu = cm.CUDAMatrix(self.G)
        self.F_gpu = cm.empty((n,r))
        self.X_gpu = cm.CUDAMatrix(self.X)
        self.GTG_gpu = cm.empty((r,r))
        self.GTGpinv_gpu = cm.empty((r,r))
        self.XG_gpu = cm.empty((n,r))
        self.XTF_gpu = cm.empty((m,r))
        self.FTF_gpu = cm.empty((r,r))
        self.XTFgreater_gpu = cm.empty((m,r))
        self.FTFgreater_gpu = cm.empty((r,r))
        self.XTFpos_gpu = cm.empty((m,r))
        self.XTFneg_gpu = cm.empty((m,r))
        self.FTFpos_gpu = cm.empty((r,r))
        self.FTFneg_gpu = cm.empty((r,r))
        self.GFTFneg_gpu = cm.empty((m,r))
        self.GFTFpos_gpu = cm.empty((m,r))

    def run(self, iterations):

        for i in range(0,iterations):
            # F = XG(G.T G)^-1
            cm.dot(self.G_gpu.T, self.G_gpu, target=self.GTG_gpu)
            try:
                self.GTGpinv_gpu = cm.CUDAMatrix(np.linalg.inv(
                                                    self.GTG_gpu.asarray()))
            except LinAlgError:
                self.GTGpinv_gpu = cm.CUDAMatrix(np.linalg.pinv(
                                                    self.GTG_gpu.asarray()))
            cm.dot(self.X_gpu, self.G_gpu, target=self.XG_gpu)
            cm.dot(self.XG_gpu, self.GTGpinv_gpu, target=self.F_gpu)

            # preparation and calculation of the matrix separations
            cm.dot(self.X_gpu.T, self.F_gpu, target=self.XTF_gpu)
            cm.dot(self.F_gpu.T, self.F_gpu, target=self.FTF_gpu)

            self.XTF_gpu.greater_than(0, target=self.XTFgreater_gpu)
            self.XTF_gpu.mult(self.XTFgreater_gpu, target=self.XTFpos_gpu)
            self.XTFpos_gpu.subtract(self.XTF_gpu, target=self.XTFneg_gpu)

            self.FTF_gpu.greater_than(0, target=self.FTFgreater_gpu)
            self.FTF_gpu.mult(self.FTFgreater_gpu, target=self.FTFpos_gpu)
            self.FTFpos_gpu.subtract(self.FTF_gpu, target=self.FTFneg_gpu)

            # compute the G update
            cm.dot(self.G_gpu, self.FTFpos_gpu, target=self.GFTFpos_gpu)
            cm.dot(self.G_gpu, self.FTFneg_gpu, target=self.GFTFneg_gpu)

            self.XTFpos_gpu.add(self.GFTFneg_gpu)
            self.XTFneg_gpu.add(self.GFTFpos_gpu)
            self.XTFpos_gpu.add_scalar(10**-9)
            self.XTFneg_gpu.add_scalar(10**-9)
            self.XTFpos_gpu.divide(self.XTFneg_gpu)
            cm.sqrt(self.XTFpos_gpu)

            self.G_gpu.mult(self.XTFpos_gpu)

            # test for convergence
            if (i % self.niter_test_conv == 0) and self.checkConvergence():
                print "NMF converged after %i iterations" % i
                break

class NMFconvex(NMFbase.NMFsemi, NMFcudamat):

    def getH(self):
        return(self.G_gpu.asarray().T)

    def getW(self):
        return(cm.dot(self.X_gpu, self.W_gpu).asarray())

    def setVariables(self):
        n, m, r = self.n, self.m, self.rank

        self.G_gpu = cm.CUDAMatrix(self.G)
        self.W_gpu = cm.CUDAMatrix(self.W)
        self.X_gpu = cm.CUDAMatrix(self.X)

        self.XTX_gpu= cm.dot(self.X_gpu.T, self.X_gpu)
        self.XTXpos_gpu = cm.empty((m,m))
        self.XTX_gpu.greater_than(0, target=self.XTXpos_gpu)
        self.XTXpos_gpu.mult(self.XTX_gpu)
        self.XTXneg_gpu = cm.empty((m,m))
        self.XTXpos_gpu.subtract(self.XTX_gpu, target=self.XTXneg_gpu)

        self.XTXnegW_gpu = cm.empty((m,r))
        self.XTXposW_gpu = cm.empty((m,r))
        self.GWT_gpu = cm.empty((m,m))
        self.update1_gpu = cm.empty((m,r))
        self.update2_gpu = cm.empty((m,r))

        self.GTG_gpu = cm.empty((r,r))
        self.XTXnegG_gpu = cm.empty((m,r))
        self.XTXposG_gpu = cm.empty((m,r))

    def run(self, iterations):

        for i in range(0,iterations):

            cm.dot(self.XTXneg_gpu, self.W_gpu, target=self.XTXnegW_gpu)
            cm.dot(self.XTXpos_gpu, self.W_gpu, target=self.XTXposW_gpu)

            # Update G
            cm.dot(self.G_gpu, self.W_gpu.T, target=self.GWT_gpu)
            # G *= np.sqrt((XTXposW + np.dot(GWT, XTXnegW))
            #              /(XTXnegW+np.dot(GWT, XTXposW)))
            cm.dot(self.GWT_gpu, self.XTXnegW_gpu, target=self.update1_gpu)
            cm.dot(self.GWT_gpu, self.XTXposW_gpu, target=self.update2_gpu)
            self.update1_gpu.add(self.XTXposW_gpu)
            self.update2_gpu.add(self.XTXnegW_gpu)
            self.update2_gpu.add_scalar(10**-9)
            self.update1_gpu.divide(self.update2_gpu)
            cm.sqrt(self.update1_gpu)
            self.G_gpu.mult(self.update1_gpu)

            # Update W
            cm.dot(self.G_gpu.T, self.G_gpu, target=self.GTG_gpu)
            #W *= np.sqrt((np.dot(XTXpos, G) + np.dot(XTXnegW, GTG))
            #                                  / (np.dot(XTXneg, G)
            #                                + np.dot(XTXposW, GTG)))
            cm.dot(self.XTXpos_gpu, self.G_gpu, target=self.XTXposG_gpu)
            cm.dot(self.XTXneg_gpu, self.G_gpu, target=self.XTXnegG_gpu)
            cm.dot(self.XTXnegW_gpu, self.GTG_gpu, target=self.update1_gpu)
            cm.dot(self.XTXposW_gpu, self.GTG_gpu, target=self.update2_gpu)
            self.update1_gpu.add(self.XTXposG_gpu)
            self.update2_gpu.add(self.XTXnegG_gpu)
            self.update2_gpu.add_scalar(10**-9)
            self.update1_gpu.divide(self.update2_gpu)
            cm.sqrt(self.update1_gpu)
            self.W_gpu.mult(self.update1_gpu)

            # test for convergence
            if (i % self.niter_test_conv == 0) and self.checkConvergence():
                print "NMF converged after %i iterations" % i
                break

class NMFKL(NMFcudamat):

    def setVariables(self):
        n, m, r = self.n, self.m, self.rank

        self.H_gpu = cm.CUDAMatrix(self.H)
        self.W_gpu = cm.CUDAMatrix(self.W)
        self.X_gpu = cm.CUDAMatrix(self.X)
        self.Wrowsum_gpu = cm.empty([r,1])
        self.WH_gpu = cm.empty([n,m])
        self.XWH_gpu = self.WH_gpu
        self.WTXWH_gpu = cm.empty([r,m])
        self.Hcolsum_gpu = cm.empty([1,r])
        self.XWHHT_gpu = cm.empty([n,r])

    def run(self, iterations):

        for i in range(0,iterations):
            #H update
            # W matrix needs to be in transposed state as else Wrowsum is in
            # the wrong shape
            self.W_gpu.set_trans(True)
            self.W_gpu.sum(1, target=self.Wrowsum_gpu)
            self.W_gpu.set_trans(False)
            cm.dot(self.W_gpu, self.H_gpu, target=self.WH_gpu)
            self.X_gpu.divide(self.WH_gpu, target=self.XWH_gpu)
            cm.dot(self.W_gpu.T, self.XWH_gpu, target=self.WTXWH_gpu)
            self.H_gpu.mult(self.WTXWH_gpu).div_by_col(self.Wrowsum_gpu)

            # W update
            self.H_gpu.set_trans(True)
            self.H_gpu.sum(0, target=self.Hcolsum_gpu)
            self.H_gpu.set_trans(False)
            cm.dot(self.W_gpu,self.H_gpu, target=self.WH_gpu)
            self.X_gpu.divide(self.WH_gpu, target=self.XWH_gpu)
            cm.dot(self.XWH_gpu, self.H_gpu.T, target=self.XWHHT_gpu)
            self.W_gpu.mult(self.XWHHT_gpu).div_by_row(self.Hcolsum_gpu)

            # test for convergence
            if (i % self.niter_test_conv == 0) and self.checkConvergence():
                print "NMF converged after %i iterations" % i
                break
