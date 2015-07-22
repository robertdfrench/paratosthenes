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
	int num_blocks_per_grid = 32;
	int num_grids = population / (num_threads_per_block * num_blocks_per_grid) + 1;
	initialize_sieve_interval<<<num_grids, num_blocks_per_grid, num_threads_per_block>>>(internal_storage, lb, population);
	state = SIS_READY;
};

__global__ void apply_sieve_device(uint64_t* storage, uint64_t storageLength, uint64_t* primes, uint64_t primesLength) 
{
	int prime_id = blockIdx.x * blockDim.x + threadIdx.x;
	int composite_id = blockIdx.y * blockDim.y + threadIdx.y;
	if (prime_id < primesLength && composite_id < storageLength) {
		uint64_t c = storage[composite_id];
		uint64_t p = primes[prime_id];
		if (c % p == 0) {
			storage[composite_id] = 0;
		}
	}
};

void CudaSearchInterval::apply_sieve(const std::vector<uint64_t>& primes, std::vector<uint64_t>& new_primes) 
{
	const uint64_t* host_primes = primes.data();
	uint64_t* device_primes;
	cudaError_t device_status;
	int num_bytes = sizeof(uint64_t) * primes.size();

	// Allocate device primes
	device_status = cudaMalloc(&device_primes, num_bytes);
	if (device_status != cudaSuccess) { state = SIS_INCONSISTENT; return;}

	// Transfer primes to device
	state = SIS_ACTIVE;
	device_status = cudaMemcpy(device_primes, host_primes, num_bytes, cudaMemcpyHostToDevice);
	if (device_status != cudaSuccess) { state = SIS_INCONSISTENT; return; }

	// Apply sieve to composite storage
	dim3 threads_per_block(32,32);
	dim3 blocks_per_grid(primes.size() / 32 + 1, population / 32 + 1);
	apply_sieve_device<<<blocks_per_grid, threads_per_block>>>(internal_storage, population, device_primes, primes.size());

	// Transfer primes back to host
	uint64_t* new_host_primes = (uint64_t*)malloc(num_bytes);
	device_status = cudaMemcpy(new_host_primes, device_primes, num_bytes, cudaMemcpyDeviceToHost);
	if (device_status != cudaSuccess) { state = SIS_INCONSISTENT; return; }

	// Sort primes
	for(uint64_t i = 0; i < population; i++) {
		uint64_t candidate = new_host_primes[i];
		if(candidate != 0) {
			new_primes.push_back(candidate);
		}
	}

	// Deallocate new host primes;
	free(new_host_primes);

	// Deallocate device primes
	device_status = cudaFree(device_primes);
	if (device_status != cudaSuccess) { state = SIS_INCONSISTENT; return; }
};
