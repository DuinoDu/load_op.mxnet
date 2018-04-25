import ctypes
_ = ctypes.CDLL('./quadratic.so')


# Revised from https://github.com/reminisce/mxnet/blob/add_op_example_for_tutorial/tests/python/unittest/test_operator.py#L4008


import unittest
import numpy as np
import mxnet as mx
import mxnet.test_utils as tu

class MySymbol(unittest.TestCase):

    def test_quadratic_forward(self):
    
        def f(x, a, b, c):
            return a * x ** 2 + b * x + c
    
        a = np.random.random_sample()
        b = np.random.random_sample()
        c = np.random.random_sample()
        for ndim in range(1, 6):
            shape = tu.rand_shape_nd(ndim, 5) 
            data = tu.rand_ndarray(shape=shape, stype='default')
            data_np = data.asnumpy()
            expected = f(data_np, a, b, c)
            output = mx.nd.contrib.quadratic_v2(data=data, a=a, b=b, c=c).asnumpy()
            tu.assert_almost_equal(output, expected)
    
    def test_quadratic_backward(self):
        a = np.random.random_sample()
        b = np.random.random_sample()
        c = np.random.random_sample()
        for ndim in range(1, 6):
            shape = tu.rand_shape_nd(ndim, 5) 
            data = tu.rand_ndarray(shape=shape, stype='default')
            data_np = data.asnumpy()

            data = mx.sym.Variable('data')
            quad_sym = mx.sym.contrib.quadratic_v2(data=data, a=a, b=b, c=c)
            #tu.check_numeric_gradient(quad_sym, [data_np])


if __name__ == '__main__':
    unittest.main()
