#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorFill.h"
#else

THC_API void THCTensor_(fill)(THCState *state, THCTensor *self, real value);
THC_API void THCTensor_(zero)(THCState *state, THCTensor *self);

#endif
