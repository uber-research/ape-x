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