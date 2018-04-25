### load_op.mxnet

Loading custom operator as an external module helps creating new operator without re-compiling mxnet from src. [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/load_op_library) and [PyTorch](https://github.com/pytorch/extension-ffi) also provide such functions.

### How-to
1. Install mxnet from src, you may refer to [script](https://github.com/DuinoDu/scripts/blob/master/shell/install/installmxnet.sh).
2. git clone https://github.com/DuinoDu/load_op.mxnet
3. cd load_op.mxnet
4. Set mxnet root path in Makefile
5. make
