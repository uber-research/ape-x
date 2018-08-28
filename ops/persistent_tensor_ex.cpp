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
