/*
Copyright (c) 2018 Uber Technologies, Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the ",Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED ",AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef TENSOR_HANDLERS_H
#define TENSOR_HANDLERS_H
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"

using namespace tensorflow;

#define TF_CALL_DATASET_TYPES(m) TF_CALL_ALL_TYPES(m) TF_CALL_QUANTIZED_TYPES(m)


namespace tensorflow {
namespace batch_util {

namespace {

template <typename T>
static Status HandleSliceToSlice(Tensor element, int64 index_element,
                                 Tensor* parent, int64 index_parent) {
  parent->flat_outer_dims<T>().chip(index_parent, 0) = element.flat_outer_dims<T>().chip(index_element, 0);
  return Status::OK();
}
}

Status CopySliceToSlice(Tensor element, int64 element_index, Tensor *parent, int64 index);
}
}

#endif