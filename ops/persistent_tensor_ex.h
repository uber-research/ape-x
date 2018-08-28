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