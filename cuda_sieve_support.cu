#include <cstdint>
#include "cuda_runtime.h"
#include "cuda.h"

__global__ void initialize_sieve_interval(uint64_t* storage, uint64_t lb, int length) {
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if (id < length) {
		storage[id] = id + lb;
	}
};
