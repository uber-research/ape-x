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

using namespace tensorflow;
using namespace std;
namespace tensorflow {

// This class avoids the need to make copies of tensors when doing enqueue_many/dequeue. When doing enqueue_many we keep track of the original Tensor.
// When attempting to use the same TensorBuffer in a new Tensor we get alignment problems.
// Copy is done during dequeue_many where we performn SliceToSlice or ElementToSlice if the element was enqueued with enqueue_many or enqueue, respectively.
class PersistentTensorEx : public PersistentTensor
{
  public:
    PersistentTensorEx() : PersistentTensor(), index_(-1) {

    }
    PersistentTensorEx(const Tensor& tensor) : PersistentTensor(tensor), index_(-1) {
        shape_ = tensor.shape();
        type_ = tensor.dtype();
    }
    PersistentTensorEx(const Tensor& tensor, int64 index) : PersistentTensor(tensor), index_(index) {
        shape_ = tensor.shape();
        shape_.RemoveDim(0);
        type_ = tensor.dtype();
    }
    virtual DataType dtype() const { return type_; }
    Status CopyToSlice(OpKernelContext* ctx, Tensor *slice, int64 index);
    Status CopyToElement(OpKernelContext* ctx, Tensor *slice);

  protected:
    int64 index_;
    TensorShape shape_;
    DataType type_;
};

} // namespace tensorflow