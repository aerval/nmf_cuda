#!/usr/bin/python
# the command line script

import argparse
import time
import numpy as np

if __name__ == "__main__":
    
    t0 = time.time()
    
    print "NMF-CUDA - a variety of update rule NMF algorithms"
    
    parser = argparse.ArgumentParser(description='A python script for running NMF on CUDA (or numpy)',
                                     epilog='Dependencies: numpy, cudamat and/or pycuda & skcuda',
                                     usage='prog <filename> [options]')
    parser.add_argument("filename", default=None,
                        help="The file of the input matrix X")
    parser.add_argument("-k", "-K", dest="rank", type=int, default=2,
                        help="Factorization Rank (default: 2)")
    parser.add_argument("-i", "-I", dest="iter", type=int, default=2000,
                        help="Maximum number of iterations")
    parser.add_argument("-j", "-J", dest="niter_test_conv", type=int, default=10,
                        help="Perform a convergence test each <niter_test_conv> iterations (default: 10). If this value is greater than <nIters> (see '-i' option), no test is performed")
    parser.add_argument("-t", "-T", dest="stop_threshold", type=int, default=40,
                        help="When matrix H has not changed on the last <stop_threshold> times that the convergence test has been performed, it is considered that the algorithm has converged to a solution and stops it.")
    parser.add_argument("-r", "-R", dest="type", default="N", type=str,
                        help="Type of NMF, P for sparse, A for affine, S for semi and C for convex else normal NMF")
    parser.add_argument("-l", "-L", dest="lib", default="n", type=str,
                        help="Base library, either just numpy, P for skcuda, C for cudamat. Skcuda does not support all algorithms.")
    parser.add_argument("-s", "-S", dest="save", default=None, type=str,
                        help="save file name, uses input name if None")
    parser.add_argument("-sh", "-sH", dest="sparseH", default=0, type=float,
                        help="Sparseness parameter of H matrix for sparse and affine NMF")
    parser.add_argument("-sw", "-sW", dest="sparseW", default=0, type=float,
                        help="Sparseness parameter of W matrix for sparse and affine NMF")
    parser.add_argument("-mh", "-mH", dest="matrixH", default=None, type=str,
                        help="Input file for H matrix")
    parser.add_argument("-mw", "-mW", dest="matrixW", default=None, type=str,
                        help="Input file for W matrix")
    parser.add_argument("-p", "-P", dest="transpose", action="store_true",
                        help="Whether the input matrix is to be computed as transpose")
    parser.add_argument("-g", "-G", dest="gpuID", default=0, type=int,
                        help="ID of the GPU, if multiple GPUs are available")
    parser.add_argument("-e", "-E", dest="encoding", default="txt",
                        help="File encoding of input and output matrices. Available options are numpy matrix format (.npy) and tab-delimitered text (.txt)")
    
    args = parser.parse_args()
    
    args.lib = args.lib.upper()
    args.type= args.type.upper()
    
    if args.encoding == "npy":
        loadMatrix = np.load
        saveMatrix = lambda filename, matrix: \
            np.save(filename, matrix.astype(np.float64).copy())
            # float64 and correct strides are necessary to provide compability
            # with RcppCNPy
    else:
        loadMatrix = np.loadtxt
        saveMatrix = np.savetxt
    
    print "Loading input matrix"
    X = loadMatrix(args.filename).astype(np.float32)
    if args.matrixH is None:
        H = None
        G = None
    else:
        H = loadMatrix(args.matrixH).astype(np.float32)
        if args.type in ["C", "S"]:
            G = H.T.copy()
    if args.matrixW is None:
        W = None
    else:
        W = loadMatrix(args.matrixW).astype(np.float32)
    
    if args.transpose:
        X = X.T.copy()  # copy is needed to keep strides in order (or in column major order)
    
    if args.save is None:
        args.save = args.filename
    
    # chose the prefered lib:
    if args.lib == "C":
        print "Using GPU %i" % args.gpuID
        import cudamat as cm
        cm.cuda_set_device(args.gpuID)
        cm.cublas_init()
        
        import NMFcudamat as nmf
        print "Chose NMFcudamat library"
    elif args.lib == "P":
        import NMFskcuda as nmf
        print "Chose NMFskcuda library"
    else:
        import NMFnumpy as nmf
        print "Chose NMFnumpy library"
    
    try:
        if args.type == "P":
            print "Running sparse NMF with sparseness constraints %f for H and %f for W" \
                % (args.sparseH, args.sparseW)
            nmfObject = nmf.NMFsparse(X, args.rank, H, W, args.sparseH,
                                      args.sparseH)
        elif args.type == "A":
            print "Running affine NMF"
            nmfObject = nmf.NMFaffine(X, args.rank, H, W, args.sparseH,
                                      args.sparseH)
        elif args.type == "S":
            print "Running semi NMF"
            nmfObject = nmf.NMFsemi(X, args.rank, G)
        elif args.type == "C":
            print "Running convex NMF"
            print "Note that convex NMF may need more iterations to converge"
            nmfObject = nmf.NMFconvex(X, args.rank, G)
        elif args.type == "KL":
            print "Running basic NMF minimizing Kullbach-Leibler Divergence"
            nmfObject = nmf.NMFKL(X, args.rank, H, W)
        else:
            print "Running basic NMF"
            nmfObject = nmf.NMF(X, args.rank, H, W)
        
        # default convergence test
        nmfObject.setConvergenceMethod("exposureChange",
                                       niter_test_conv = args.niter_test_conv,
                                       stop = args.stop_threshold)
        
        # run the actual thing
        nmfObject.start(args.iter)
        
        saveMatrix(args.save + "_H." + args.encoding, nmfObject.getH())
        saveMatrix(args.save + "_W." + args.encoding, nmfObject.getW())
        
        print "Distance: " + str(nmfObject.frobError())
    
    except AttributeError:
        raise Exception("The chosen NMF algorithm is not implemented in the chosen library")
    
    t1 = time.time()
    print "Time taken by NMF-CUDA: ", t1-t0
        
