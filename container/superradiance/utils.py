from tensorflow.python.ops import array_ops
import numpy as np

def kronecker_product(mat1, mat2):
    """Computes the Kronecker product two 2D tf-tensors."""
    m1, n1 = mat1.get_shape().as_list()
    mat1_rsh = array_ops.reshape(mat1, [m1, 1, n1, 1])
    m2, n2 = mat2.get_shape().as_list()
    mat2_rsh = array_ops.reshape(mat2, [1, m2, 1, n2])
  
    return array_ops.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2])

def SparseIndices(X):
    """Retrieves indices, values, and dense_shape from a numpy sparse matrix"""
    indicesT = X.nonzero()
    indices = np.transpose(indicesT)
    values = X.data
    denseshape = X.shape
    
    return {'indices':indices, 'values':values, 'dense_shape':denseshape}