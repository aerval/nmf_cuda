
import ctypes
# need to explicitly import libgomp for some reason, autoimport in new version
ctypes.CDLL('libgomp.so.1', mode=ctypes.RTLD_GLOBAL)

import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.elementwise as el
from pycuda import driver, compiler, gpuarray, tools
import numpy as np
from skcuda import linalg
from skcuda import misc
linalg.init()
import NMFbase

class NMFskcuda(NMFbase.NMFbase):

    def getH(self):
        return(self.H_gpu.get())

    def getW(self):
        return(self.W_gpu.get())

# code for the updating steps (e.g H *= WTX / WTWH)
# => elementwise multiply and divide
update_kernel_code = """
__global__ void ew_md(float* a, float* b, float* c) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < %i)
      a[idx] *= b[idx] / c[idx];
}"""

class NMF(NMFskcuda):

    def setVariables(self):

        # compile the update functions for H and W as elementwise Matrix-Mult.
        # is not in skcuda
        H_size = self.rank * self.m
        W_size = self.n * self.rank
        max_threads = tools.DeviceData().max_threads
        self.block_H = int(np.min([H_size, max_threads]))
        self.block_W = int(np.min([W_size, max_threads]))
        self.grid_H = np.int(np.ceil(H_size/np.float32(self.block_H)))
        self.grid_W = np.int(np.ceil(W_size/np.float32(self.block_W)))
        mod_H = compiler.SourceModule(update_kernel_code % H_size)
        mod_W = compiler.SourceModule(update_kernel_code % W_size)
        self.update_H = mod_H.get_function("ew_md")
        self.update_W = mod_W.get_function("ew_md")

        # allocate the matrices on the GPU
        self.H_gpu = gpuarray.to_gpu(self.H)
        self.W_gpu = gpuarray.to_gpu(self.W)
        self.X_gpu = gpuarray.to_gpu(self.X)
        self.WTW_gpu = gpuarray.empty((self.rank, self.rank), np.float32)
        self.WTWH_gpu = gpuarray.empty(self.H.shape, np.float32)
        self.WTX_gpu = gpuarray.empty(self.H.shape, np.float32)
        self.XHT_gpu = gpuarray.empty(self.W.shape, np.float32)
        self.WH_gpu = gpuarray.empty(self.X.shape, np.float32)
        self.WHHT_gpu = gpuarray.empty(self.W.shape, np.float32)

    def run(self, iterations):

        for i in range(0,iterations):
            # update H
            linalg.add_dot(self.W_gpu, self.X_gpu, self.WTX_gpu, transa="T",
                           beta=0.) # add_dot is faster than dot, dot calls ad
            linalg.add_dot(self.W_gpu, self.W_gpu, self.WTW_gpu, transa="T",
                           beta=0.)
            linalg.add_dot(self.WTW_gpu, self.H_gpu, self.WTWH_gpu, beta=0.)
            self.update_H(self.H_gpu, self.WTX_gpu, self.WTWH_gpu,
                          block=(self.block_H, 1, 1), grid=(self.grid_H, 1))

            # update W
            linalg.add_dot(self.X_gpu, self.H_gpu, self.XHT_gpu, transb="T",
                           beta=0.)
            linalg.add_dot(self.W_gpu, self.H_gpu, self.WH_gpu, beta=0.)
            linalg.add_dot(self.WH_gpu, self.H_gpu, self.WHHT_gpu, transb="T", beta=0.)
            self.update_W(self.W_gpu, self.XHT_gpu, self.WHHT_gpu,
                          block=(self.block_W, 1, 1), grid=(self.grid_W, 1))

            # test for convergence
            if (i % self.niter_test_conv == 0) and self.checkConvergence():
                print "NMF converged after %i iterations" % i
                break

# code for dividing a matrix in positive and negative parts
# and the G update step
matrix_separation_code = """
#include "math.h"

__global__ void matrix_separation(float* input, float* positive,
                                  float* negative) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < %i) {
        if (input[idx] > 0.) {
            positive[idx] = input[idx];
            negative[idx] = 0.;
        } else {
            positive[idx] = 0.;
            negative[idx] = - input[idx];
        }
    }
}

"""

G_update_code = """
__global__ void G_ew_update(float* G, float* XTFpos, float* GFTFneg,
                            float* XTFneg, float* GFTFpos) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < %i) {
        G[idx] *= sqrtf( (XTFpos[idx] + GFTFneg[idx] + 1e-9)
                         / (XTFneg[idx] + GFTFpos[idx] + 1e-9) );
    }
}"""

class NMFsemi(NMFbase.NMFsemi, NMFskcuda):

    def getH(self):
        return(self.G_gpu.get().T)

    def getW(self):
        return(self.F_gpu.get())

    def setVariables(self):
        n, m, r = self.n, self.m, self.rank

        # compile the matrix separations and G update functions for CUDA
        G_size = m * r
        FTF_size = r**2
        max_threads = tools.DeviceData().max_threads
        self.block_G = int(np.min([G_size, max_threads]))
        self.grid_G = np.int(np.ceil(G_size/np.float32(self.block_G)))
        self.block_FTF = int(np.min([FTF_size, max_threads]))
        self.grid_FTF = np.int(np.ceil(FTF_size/np.float32(self.block_FTF)))
        mod_msepXTF = compiler.SourceModule(matrix_separation_code % G_size)
        mod_msepFTF = compiler.SourceModule(matrix_separation_code % FTF_size)
        mod_Gupdate = compiler.SourceModule(G_update_code % G_size)
        self.matrix_separationXTF = \
             mod_msepXTF.get_function("matrix_separation")
        self.matrix_separationFTF = \
             mod_msepFTF.get_function("matrix_separation")
        self.G_ew_update = mod_Gupdate.get_function("G_ew_update")

        # allocate the matrices on the GPU
        self.G_gpu = gpuarray.to_gpu(self.G)
        self.F_gpu = gpuarray.empty((n,r), np.float32)
        self.X_gpu = gpuarray.to_gpu(self.X)
        self.GTG_gpu = gpuarray.empty((r,r), np.float32)
        self.GTGinv_gpu = gpuarray.empty((r,r), np.float32)
        self.XG_gpu = gpuarray.empty((n,r), np.float32)
        self.XTF_gpu = gpuarray.empty((m,r), np.float32)
        self.FTF_gpu = gpuarray.empty((r,r), np.float32)
        self.XTFpos_gpu = gpuarray.empty((m,r), np.float32)
        self.XTFneg_gpu = gpuarray.empty((m,r), np.float32)
        self.FTFpos_gpu = gpuarray.empty((r,r), np.float32)
        self.FTFneg_gpu = gpuarray.empty((r,r), np.float32)
        self.GFTFneg_gpu = gpuarray.empty((m,r), np.float32)
        self.GFTFpos_gpu = gpuarray.empty((m,r), np.float32)

    def run(self, iterations):

        for i in range(0,iterations):
            # F = XG(G.T G)^-1
            linalg.add_dot(self.G_gpu, self.G_gpu, self.GTG_gpu, transa="T",
                           beta=0.)
            try:
                self.GTGinv_gpu.set(np.linalg.inv(self.GTG_gpu.get()))
                # linalg.pinv only worked with CULA
            except LinAlgError:
                self.GTGinv_gpu.set(np.linalg.iinv(self.GTG_gpu.get()))
            linalg.add_dot(self.X_gpu, self.G_gpu, self.XG_gpu, beta=0.)
            linalg.add_dot(self.XG_gpu, self.GTGinv_gpu, self.F_gpu, beta=0.)

            # preparation and calculation of the matrix separations
            linalg.add_dot(self.X_gpu, self.F_gpu, self.XTF_gpu, transa="T",
                           beta=0.)
            linalg.add_dot(self.F_gpu, self.F_gpu, self.FTF_gpu, transa="T",
                           beta=0.)
            self.matrix_separationXTF(self.XTF_gpu, self.XTFpos_gpu,
                                      self.XTFneg_gpu,
                                      block=(self.block_G, 1, 1),
                                      grid=(self.grid_G, 1))
            self.matrix_separationFTF(self.FTF_gpu, self.FTFpos_gpu,
                                      self.FTFneg_gpu,
                                      block=(self.block_FTF, 1, 1),
                                      grid=(self.grid_FTF, 1))

            # compute the G update
            linalg.add_dot(self.G_gpu, self.FTFpos_gpu, self.GFTFpos_gpu,
                           beta=0.)
            linalg.add_dot(self.G_gpu, self.FTFneg_gpu, self.GFTFneg_gpu,
                           beta=0.)
            self.G_ew_update(self.G_gpu, self.XTFpos_gpu, self.GFTFneg_gpu,
                             self.XTFneg_gpu, self.GFTFpos_gpu,
                             block=(self.block_G, 1, 1),
                             grid=(self.grid_G, 1))

            # test for convergence
            if (i % self.niter_test_conv == 0) and self.checkConvergence():
                print "NMF converged after %i iterations" % i
                break
