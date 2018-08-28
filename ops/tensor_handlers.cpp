/*
Copyright (c) 2018 Uber Technologies, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <iostream>
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensor_handlers.h"

using namespace tensorflow;
using namespace std;

#define TF_CALL_DATASET_TYPES(m) TF_CALL_ALL_TYPES(m) TF_CALL_QUANTIZED_TYPES(m)


namespace tensorflow {
namespace batch_util {

namespace {

Status ValidateInput(const Tensor& parent, int64 index, const Tensor& element, int64 index2) {
  DCHECK_NE(parent.dim_size(0), 0);
  DCHECK_NE(element.dim_size(0), 0);
  DCHECK_GE(index, 0);
  DCHECK_GE(index2, 0);
  if ((element.NumElements() / element.dim_size(0)) != (parent.NumElements() / parent.dim_size(0))) {
    TensorShape chip_shape = element.shape();
    chip_shape.RemoveDim(0);
    TensorShape chip_shape2 = parent.shape();
    chip_shape2.RemoveDim(0);
    return errors::Internal(
        "ValidateInput Cannot perform copy: number of elements does not match. "
        " Shapes are: [element]: ",
        chip_shape.DebugString(),
        ", [parent slice]: ", chip_shape2.DebugString());
  }
  return Status::OK();
}

}


Status CopySliceToSlice(Tensor element, int64 element_index, Tensor* parent, int64 index) {
  TF_RETURN_IF_ERROR(ValidateInput(*parent, index, element, element_index));
#define HANDLE_TYPE(T)                                                \
  case DataTypeToEnum<T>::value: {                                    \
    return HandleSliceToSlice<T>(std::move(element), element_index, parent, index);\
  }

  switch (element.dtype()) {
    TF_CALL_ALL_TYPES(HANDLE_TYPE);
    TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented("CopySliceToSlice Unhandled data type: ",
                                   element.dtype());
  }
}
}
}