#include "cuda_search_interval.h"

#include <cstdint>

#include "cuda_runtime.h"
#include "cuda.h"

__global__ void initialize_sieve_interval(uint64_t* storage, uint64_t lb, int length) 
{
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if (id < length) {
		storage[id] = id + lb;
	}
};

void CudaSearchInterval::initialize() 
{
	int num_threads_per_block = 32;
	int num_blocks_per_grid = 33;
	int num_grids = population / (num_threads_per_block * num_blocks_per_grid) + 1;
	initialize_sieve_interval<<<num_grids, num_blocks_per_grid, num_threads_per_block>>>(internal_storage, lb, population);
	state = SIS_READY;
};

__global__ void apply_sieve(uint64_t* storage, uint64_t storageLength, uint64_t* primes, uint64_t primesLength) 
{

};

void CudaSearchInterval::apply_sieve() 
{
	state = SIS_ACTIVE;
}
