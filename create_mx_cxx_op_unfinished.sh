#!/bin/bash

if [ ! -n "$1" ];then
    echo "create_mx_cpp_op [op name, eg: quadratic]"
    exit 1
fi

if [ -d $1 ];then
    echo "$1 already exists"
    exit 1
fi
mkdir $1 && cd $1

touch ${1}_op-inl.h 
touch ${1}_op.cc
touch ${1}_op.cu

function write() {
    echo "$1" >> $cur
}
op_name_upper=`echo $1 | tr '[:lower:]' '[:upper:]'`

#1. Define the parameter struct for registering `a`, `b`, and `c` in `quadratic_op-inl.h`.
#2. Define type and shape inference functions in `quadratic_op-inl.h`.
#3. Define forward and backward functions in `quadratic_op-inl.h`.

cur=${1}_op-inl.h
write \
"#ifndef MXNET_OPERATOR_CONTRIB_${op_name_upper}_INL_H_
#define MXNET_OPERATOR_CONTRIB_${op_name_upper}_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/base.h>
#include <nnvm/tuple.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <valarray>
#include \"../operator_common.h\"

namespace mxnet {
namespace op {

struct ${1^}Param : public dmlc::Parameter<${1^}Param> {
    float a;
    DMLC_DECLARE_PARAMETER(${1^}Param) {
        DMLC_DECLARE_FIELD(a)
            .set_default(0.0)
            .describe(\"description text\");
    }
};

inline bool ${1^}OpShape(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape>* in_attrs,
                         std::vector<TShape>* out_attrs){
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);

    SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
    return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
}

inline bool ${1^}OpType(const nnvm::NodeAttrs& attrs,
                        std::vector<int>* in_attrs,
                        std::vector<int>* out_attrs){
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);

    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
    return out_attrs->at(0) != -1; 
}

template<int req>
struct ${1}_forward {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                    const float a) {

        KERNEL_ASSIGN(out_data[i], req, in_data[i] * in_data[i] * a); // TODO 
    }
};

template<int req>
struct ${1}_backward {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType* in_grad, const DType* out_grad,
                                    const DType* in_data, const float a) {
        KERNEL_ASSIGN(in_grad[i], req, out_grad[i] * ( 2 * a * in_data[i] )); // TODO
    }
};

template<typename xpu>
void ${1^}OpForward(const nnvm::NodeAttr& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    const TBlob& in_data = inputs[0];
    const TBlob& out_data = outputs[0];
    const ${1^}Param& param = nnvm::get<${1^}Param>(attrs.parsed);
    using namespace mxnet_op;
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
        MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
            Kernel<${1}_forward<req_type>, xpu>::Launch(
                s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
                param.a);   
        });
    });
}


template<typename xpu>
void ${1^}OpBackward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_grad = inputs[0];
  const TBlob& in_data = inputs[1];
  const TBlob& in_grad = outputs[0];
  const ${1^}Param& param = nnvm::get<${1^}Param>(attrs.parsed);
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<${1}_backward<req_type>, xpu>::Launch(
          s, in_grad.Size(), in_grad.dptr<DType>(), out_grad.dptr<DType>(),
          in_data.dptr<DType>(), param.a);
    });
  });
}

} // namespace op
} // namespace mxnet
"

#4. Register the operator using [nnvm](https://github.com/dmlc/nnvm)
#   in `quadratic_op.cc` and `quadratic_op.cu` for
#   CPU and GPU computing, respectively.

cur=${1}_op.cc
write \
"#include \"./$1_op-inl.h\"

namespace mxnet{
namespace op{

DMLC_REGISTER_PARAMETER(${1^}Param);

NNVM_REGISTER_OP($1)
.describe(R\"code(#TODO)code\"ADD_FILELINE)
.set_attr_parser(ParamParser<${1^}Param>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>


}
}

"



