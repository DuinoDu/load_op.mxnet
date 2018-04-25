mxnet_root = ${HOME}/src/incubator-mxnet
dmlc_inc = ${mxnet_root}/3rdparty/dmlc-core/include
nnvm_inc = ${mxnet_root}/3rdparty/nnvm/include
mshadow_inc = ${mxnet_root}/3rdparty/mshadow
dlpack_inc = ${mxnet_root}/3rdparty/dlpack/include
mxnet_src_inc = ${mxnet_root}/src/operator
mxnet_inc = ${mxnet_root}/include
cuda_inc=/usr/local/cuda/include

mxnet_lib = ${mxnet_root}/lib

INC = -I${dmlc_inc} -I${nnvm_inc} -I${mxnet_inc} -I${mshadow_inc} -I${dlpack_inc} -I${cuda_inc} -I${mxnet_src_inc} 
DEFINE = -D MSHADOW_USE_CBLAS -D MSHADOW_USE_CUDA -D MSHADOW_USE_CUDNN=1 -D MSHADOW_USE_CUSOLVER=1
LIB = -L${mxnet_lib} -lmxnet


all: clean build cleancache test

build: quadratic.so

quadratic_op_cu.o: quadratic_op.cu
	nvcc -std=c++11 -c -o $@ $? -O2 -x cu -Xcompiler -fPIC ${INC} ${DEFINE}

quadratic.so: quadratic_op.cc quadratic_op_cu.o
	g++ -std=c++11 -shared -o $@ $? -O2 -fPIC ${INC} ${DEFINE} ${LIB}

clean:
	rm -f *.o *.so *.pyc

cleancache:
	rm -f *.o

test:
	python test_quadratic.py
