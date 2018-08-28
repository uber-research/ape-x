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

#ifndef TF_ENV_H_
#define TF_ENV_H_
#include <iostream>
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
class BaseEnvironment : public ResourceBase
{
  public:
    virtual bool is_done(int idx) = 0;
    virtual void reset(int i, int numNoops = 0, int maxFrames = 100000) = 0;
};

template<typename T>
class StepInterface
{
  public:
    virtual TensorShape get_action_shape() = 0;
    virtual float step(int idx, const T* action) = 0;
};

template<typename T>
class Environment : public BaseEnvironment
{
  public:
    virtual void get_observation(T* data, int idx) = 0;
    virtual TensorShape get_observation_shape() = 0;
};

class EnvironmentMakeOp : public OpKernel {
  public:
    explicit EnvironmentMakeOp(OpKernelConstruction *context);

    // The resource is deleted from the resource manager only when it is private
    // to kernel. Ideally the resource should be deleted when it is no longer held
    // by anyone, but it would break backward compatibility.
    virtual ~EnvironmentMakeOp() override;

    void Compute(OpKernelContext *context) override LOCKS_EXCLUDED(mu_);

  protected:
    // Variables accessible from subclasses.
    tensorflow::mutex mu_;
    ContainerInfo cinfo_ GUARDED_BY(mu_);
    BaseEnvironment* resource_ GUARDED_BY(mu_) = nullptr;
    int batch_size;

  private:
    // During the first Compute(), resource is either created or looked up using
    // shared_name. In the latter case, the resource found should be verified if
    // it is compatible with this op's configuration. The verification may fail in
    // cases such as two graphs asking queues of the same shared name to have
    // inconsistent capacities.
    virtual Status VerifyResource(BaseEnvironment *resource);

    PersistentTensor handle_ GUARDED_BY(mu_);

    virtual Status CreateResource(OpKernelContext *context, BaseEnvironment **ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

    TF_DISALLOW_COPY_AND_ASSIGN(EnvironmentMakeOp);
};

#endif
