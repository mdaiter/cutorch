// FIXME: I don't know if we should do it this way, but there should
// only be one def of this. Should this be in the non-generic dir
// instead, along with other unique defs?
#ifndef THC_TENSOR_FILL_GENERIC_DEFS
#define THC_TENSOR_FILL_GENERIC_DEFS

template <typename T>
struct TensorFillOp {
  TensorFillOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* v) { *v = val; }

  const T val;
};

// FIXME: hack to deal with `half`
template <typename IN, typename OUT>
struct ValueConverter {
  static OUT get(IN v) { return v; }
};

template <>
struct ValueConverter<half, short> {
  static short get(half v) { return v.x; }
};

#endif

#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorFill.cu"
#else

THC_API void
THCTensor_(fill)(THCState* state, THCTensor *self_, real value)
{
  THAssert(THCTensor_(checkGPU)(state, 1, self_));

  typename TensorUtils<THCTensor>::DataType v =
    ValueConverter<real,
                   typename TensorUtils<THCTensor>::DataType>::get(value);

  if (!cutorch_pointwiseApply1(
        state, self_,
        // FIXME: `real` should be sufficient, but code can't handle
        // half at the moment. In any case, this is more C++-ish
        TensorFillOp<typename TensorUtils<THCTensor>::DataType>(v))) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(zero)(THCState *state, THCTensor *self_)
{
  THAssert(THCTensor_(checkGPU)(state, 1, self_));
  if (THCTensor_(isContiguous)(state, self_)) {
    THCudaCheck(cudaMemsetAsync(THCTensor_(data)(state, self_),
                                0,
                                sizeof(real) * THCTensor_(nElement)(state, self_),
                                THCState_getCurrentStream(state)));
  } else {
    if (!cutorch_pointwiseApply1(
          state, self_,
          // FIXME: `real` should be sufficient, but code can't handle
          // half at the moment. In any case, this is more C++-ish
          TensorFillOp<typename TensorUtils<THCTensor>::DataType>(
            (typename TensorUtils<THCTensor>::DataType) 0))) {
      THArgCheck(false, 1, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

#endif
