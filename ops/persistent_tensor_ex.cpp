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

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "persistent_tensor_ex.h"
#include "tensorflow/core/util/batch_util.h"
#include "tensor_handlers.h"

using namespace tensorflow;
using namespace std;


Status PersistentTensorEx::CopyToSlice(OpKernelContext* ctx, Tensor* slice, int64 index) {
    if(index_ == -1)
        // Copy element
        return batch_util::CopyElementToSlice(*AccessTensor(ctx), slice, index);
    return batch_util::CopySliceToSlice(*AccessTensor(ctx), index_, slice, index);
}

Status PersistentTensorEx::CopyToElement(OpKernelContext* ctx, Tensor* element) {
    if(index_ == -1) {
        // Copy element
        *element = *AccessTensor(ctx);
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        type_, shape_, element));
    return batch_util::CopySliceToElement(*AccessTensor(ctx), element, index_);
}
