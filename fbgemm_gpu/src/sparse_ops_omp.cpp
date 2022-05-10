/*
 * Copyright (c) 2022 AMD
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This source file is an example of how to replace cuda kernels with OpenMP
// target offload functions. Please read the comment blocks below for
// more information on how this is done.

#include <omp.h>

#ifdef __CUDA__
#error Only for OpenMP.   Cannot compile this source with a CUDA compiler
#endif

// Carefully add necessary headers here. By convention, Source files, like
// this, ending in "_omp.cpp" go through host and device passes of the
// OpenMP target offload compiler. Some pytorch headers expect certain
// compile conditions on the device pass. For processing those headers,
// we temporarily fake a cuda or hip compiler by setting certain macros.
// Like the CUDA and HIP compilers, the OpenMP clang compiler sets macro
// __CUDA_ARCH__ ONLY during the device pass. So we test that macro to
// setup the fake environment. We do not want to do this on the host pass.
#ifdef __CUDA_ARCH__
#pragma push_macro("__CUDACC__")
#pragma push_macro("__device__")
#pragma push_macro("__host__")
#define __FAKING_CUDA_HIP_ENV__
#if defined(__NVPTX__)
#define __CUDACC__
#include <vector_functions.h>
#endif
#if defined(__AMDGCN__)
#define __HIPCC__
#endif
// #define __CUDA__ maybe we need to set __CUDA__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#define C10_USING_CUSTOM_GENERATED_MACROS
#endif

#include "fbgemm_gpu/sparse_ops_utils.h"
#include <ATen/CPUFunctions.h>

#if defined(__FAKING_CUDA_HIP_ENV__)
#undef __FAKING_CUDA_HIP_ENV__
#pragma pop_macro("__CUDACC__")
#pragma pop_macro("__device__")
#pragma pop_macro("__host__")
#endif

using Tensor = at::Tensor;

namespace fbgemm_gpu {

// This OpenMP target offload kernel is nearly the same as the cpu source
// reorder_batched_ad_lengths_ function with the exception of the added
// OpenMP target pragma and the manual loop collapse. Instead of OpenMP
// target maps, we use is_device_ptr because cuda manages all the device data.
template <typename index_t, typename scalar_t>
void reorder_batched_ad_lengths_omp(const Tensor &cat_ad_lengths,
                                    const Tensor &batch_offsets,
                                    const int64_t num_ads_in_batch,
                                    Tensor &output) {
  const int64_t nB = batch_offsets.numel() - 1;
  const int64_t nT = cat_ad_lengths.numel() / num_ads_in_batch;
  const auto *batch_offsets_data = batch_offsets.data_ptr<index_t>();
  const auto *cat_ad_lengths_data = cat_ad_lengths.data_ptr<scalar_t>();
  auto *output_data = output.data_ptr<scalar_t>();
#pragma omp target teams distribute parallel for is_device_ptr(                \
    cat_ad_lengths_data, batch_offsets_data, output_data)
  for (auto k = 0; k < nB * nT; k++) {
    auto b = k / nT;
    auto t = k % nT;
    const auto num_ads_b = batch_offsets_data[b + 1] - batch_offsets_data[b];
    const int32_t input_segment_start =
        nT * batch_offsets_data[b] + t * num_ads_b;
    const int32_t output_segment_start =
        t * num_ads_in_batch + batch_offsets_data[b];
    for (auto i = 0; i < num_ads_b; i++) {
      output_data[output_segment_start + i] =
          cat_ad_lengths_data[input_segment_start + i];
    }
  }
}

// The omp wrapper function must have the same name and interface as the _gpu
// wrapper functions found in sparse_ops.cu. To avoid duplicate names, the _gpu
// functions in the .cu file are ifdef'd out with ifndef _FBGEMM_OMP_TARGET.
// This _omp.cpp file is only compiled with --omp_target. See CMakeLists.txt.
// The device data checks are also the same because CUDA is managing all the
// device data. However the DISPATCH is to the omp function above.
Tensor reorder_batched_ad_lengths_gpu(const Tensor &cat_ad_lengths,
                                      const Tensor &batch_offsets,
                                      const int64_t num_ads_in_batch) {
  /* These from .cu */
  TENSOR_ON_CUDA_GPU(cat_ad_lengths);
  TENSOR_ON_CUDA_GPU(batch_offsets);
  TENSORS_ON_SAME_DEVICE(cat_ad_lengths, batch_offsets);

  Tensor reordered_cat_ad_lengths = at::empty_like(cat_ad_lengths);
  AT_DISPATCH_INDEX_TYPES(
      batch_offsets.scalar_type(), "reorder_batched_ad_lengths_gpu_kernel1",
      [&] {
        AT_DISPATCH_ALL_TYPES(
            cat_ad_lengths.scalar_type(),
            "reorder_batched_ad_lengths_gpu_kernel2", [&] {
              reorder_batched_ad_lengths_omp<index_t, scalar_t>(
                  cat_ad_lengths, batch_offsets, num_ads_in_batch,
                  reordered_cat_ad_lengths);
            });
      });
  return reordered_cat_ad_lengths;
}

template <typename index_t, typename scalar_t>
void reorder_batched_ad_indices_omp(const Tensor &cat_ad_offsets,
                                    const Tensor &cat_ad_indices,
                                    const Tensor &reordered_cat_ad_offsets,
                                    const Tensor &batch_offsets,
                                    const int64_t num_ads_in_batch,
                                    Tensor &output) {
  const int64_t nB = batch_offsets.numel() - 1;
  const int64_t nT = (cat_ad_offsets.numel() - 1) / num_ads_in_batch;
  const auto *batch_offsets_data = batch_offsets.data_ptr<int32_t>();
  const auto *cat_ad_offsets_data = cat_ad_offsets.data_ptr<index_t>();
  const auto *reordered_cat_ad_offsets_data =
      reordered_cat_ad_offsets.data_ptr<index_t>();
  const auto *cat_ad_indices_data = cat_ad_indices.data_ptr<scalar_t>();
  auto *output_data = output.data_ptr<scalar_t>();

#pragma omp target teams distribute parallel for is_device_ptr(                \
    batch_offsets_data, cat_ad_offsets_data, reordered_cat_ad_offsets_data,    \
    cat_ad_indices_data, output_data)
  for (auto k = 0; k < nB * nT; k++) {
    auto b = k / nT;
    auto t = k % nT;
    const auto num_ads_b = batch_offsets_data[b + 1] - batch_offsets_data[b];
    const int32_t input_segment_offset_start =
        nT * batch_offsets_data[b] + t * num_ads_b;
    const int32_t input_segment_offset_end =
        nT * batch_offsets_data[b] + t * num_ads_b + num_ads_b;

    const auto input_segment_start =
        cat_ad_offsets_data[input_segment_offset_start];
    const auto input_segment_end =
        cat_ad_offsets_data[input_segment_offset_end];

    const auto output_segment_offset_start =
        t * num_ads_in_batch + batch_offsets_data[b];
    const auto output_segment_start =
        reordered_cat_ad_offsets_data[output_segment_offset_start];

    for (auto i = 0; i < input_segment_end - input_segment_start; i++) {
      output_data[output_segment_start + i] =
          cat_ad_indices_data[input_segment_start + i];
    }
  }
}

// Must be same name and interface found in sparse_ops.cu
Tensor reorder_batched_ad_indices_gpu(const Tensor &cat_ad_offsets,
                                      const Tensor &cat_ad_indices,
                                      const Tensor &reordered_cat_ad_offsets,
                                      const Tensor &batch_offsets,
                                      const int64_t num_ads_in_batch) {
  TENSOR_ON_CUDA_GPU(cat_ad_offsets);
  TENSOR_ON_CUDA_GPU(cat_ad_indices);
  TENSOR_ON_CUDA_GPU(reordered_cat_ad_offsets);
  TENSOR_ON_CUDA_GPU(batch_offsets);
  TENSORS_ON_SAME_DEVICE(cat_ad_offsets, cat_ad_indices);
  TENSORS_ON_SAME_DEVICE(cat_ad_offsets, reordered_cat_ad_offsets);
  TENSORS_ON_SAME_DEVICE(cat_ad_offsets, batch_offsets);

  Tensor reordered_cat_ad_indices = at::empty_like(cat_ad_indices);
  AT_DISPATCH_INDEX_TYPES(
      cat_ad_offsets.scalar_type(), "reorder_batched_ad_indices_gpu_kernel_1",
      [&] {
        AT_DISPATCH_ALL_TYPES(
            cat_ad_indices.scalar_type(),
            "reorder_batched_ad_indices_gpu_kernel_2", [&] {
              reorder_batched_ad_indices_omp<index_t, scalar_t>(
                  cat_ad_offsets, cat_ad_indices, reordered_cat_ad_offsets,
                  batch_offsets, num_ads_in_batch, reordered_cat_ad_indices);
            });
      });

  return reordered_cat_ad_indices;
}

} // end of namespace fbgemm_gpu
