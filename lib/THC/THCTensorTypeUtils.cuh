#ifndef THC_TENSOR_TYPE_UTILS_INC
#define THC_TENSOR_TYPE_UTILS_INC

#include <cuda.h>
#include <assert.h>
#include "THCGeneral.h"
#include "THCTensor.h"
#include "THCTensorInfo.cuh"

/// A utility for accessing THCuda*Tensor types in a generic manner

template <typename TensorType>
struct TensorUtils {
};

#define TENSOR_UTILS(TENSOR_TYPE, DATA_TYPE)                            \
  template <>                                                           \
  struct TensorUtils<TENSOR_TYPE> {                                     \
    typedef DATA_TYPE DataType;                                         \
                                                                        \
    static TENSOR_TYPE* newContiguous(THCState* state, TENSOR_TYPE* t); \
    static void free(THCState* state, TENSOR_TYPE* t);                  \
    static DATA_TYPE* getData(THCState* state, TENSOR_TYPE* t);         \
    static long getNumElements(THCState* state, TENSOR_TYPE* t);        \
    static long getSize(THCState* state, TENSOR_TYPE* t, int dim);      \
    static long getStride(THCState* state, TENSOR_TYPE* t, int dim);    \
    static int getDims(THCState* state, TENSOR_TYPE* t);                \
    static bool isContiguous(THCState* state, TENSOR_TYPE* t);          \
    static void copyIgnoringOverlaps(THCState* state,                   \
                                     TENSOR_TYPE* dst, TENSOR_TYPE* src); \
    /* Determines if the given tensor has overlapping data points (i.e., */ \
    /* is there more than one index into the tensor that references */  \
    /* the same piece of data)? */                                      \
    static bool overlappingIndices(THCState* state, TENSOR_TYPE* t);    \
    /* Can we use 32 bit math for indexing? */                          \
    static bool canUse32BitIndexMath(THCState* state, TENSOR_TYPE* t);  \
  }

TENSOR_UTILS(THCudaByteTensor, unsigned char);
TENSOR_UTILS(THCudaCharTensor, char);
TENSOR_UTILS(THCudaShortTensor, short);
TENSOR_UTILS(THCudaIntTensor, int);
TENSOR_UTILS(THCudaLongTensor, long);
TENSOR_UTILS(THCudaTensor, float);
TENSOR_UTILS(THCudaDoubleTensor, double);

#if CUDA_VERSION >= 7050
// FIXME: should be half, but requires conversion functors and CPU vs
// GPU types (I think?)
TENSOR_UTILS(THCudaHalfTensor, short);
#endif

#undef TENSOR_UTILS

template <typename TensorType, typename IndexType>
TensorInfo<typename TensorUtils<TensorType>::DataType, IndexType>
getTensorInfo(THCState* state, TensorType* t) {
  printf("getTensorInfo\n");
  IndexType sz[MAX_CUTORCH_DIMS];
  IndexType st[MAX_CUTORCH_DIMS];

  int dims = TensorUtils<TensorType>::getDims(state, t);
  for (int i = 0; i < dims; ++i) {
    sz[i] = TensorUtils<TensorType>::getSize(state, t, i);
    st[i] = TensorUtils<TensorType>::getStride(state, t, i);
  }

  return TensorInfo<typename TensorUtils<TensorType>::DataType, IndexType>(
    TensorUtils<TensorType>::getData(state, t), dims, sz, st);
}

#endif // THC_TENSOR_TYPE_UTILS_INC
