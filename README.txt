This repository holds the files necessary to test a crude
CUDA-accelerated linear 2D finite element based laplacian
operator assembly routine.

The main driver program is "laplace_test.m"

The driver program reads a finite element mesh contained in 
"ex_5_9_2.msh" and loads the data into a MATLAB structure.

The driver then runs two functions that are written to take the
mesh structure and construct a 2D linear finite element Laplacian
operator using 3-node linear triangular elements.  

The first function accomplishes this task using  MATLAB's native M-language
only.  The second function accomplishes this using a compiled C/C++ program
that uses CUDA on NVIDIA GPUs to do the matrix assembly which turns out to
be much faster albiet in single precision arithmetic that seems to accumulate
a lot of error.  (sorry) 

The goal was to test the FEM operator assembly routine with CUDA acceleration. 

As written, the bulk of the time for the CUDA version is taken up writing
an input file to disk describing the element geometry.  Actual matrix
assembly takes up a small fraction of that time.

The results of this code and the experience gained writing it are behind
a not-so-recent blog post which can be viewed at:
http://http://noiceinmyscotchplease.blogspot.com/2011/12/cuda-accelerated-sparse-matrix-assembly.html
