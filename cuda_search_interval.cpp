#include "cuda_search_interval.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <string>

#include "cuda_runtime.h"
#include "cuda.h"

#define COMPOSITE 0
#define MAX_STORAGE_SIZE 1024 * 1024 * 100

void CudaSearchInterval::set_extrema_safely(uint64_t lowerbound, uint64_t upperbound) {
	uint64_t potential_population = upperbound - lowerbound;
	if (potential_population > MAX_STORAGE_SIZE) {
		upperbound = lowerbound + MAX_STORAGE_SIZE - 1;
	}
	
	lb = lowerbound;
	ub = upperbound;
	capacity = MAX_STORAGE_SIZE;
	population = ub - lb;
	state = SIS_INCONSISTENT;
};

CudaSearchInterval::CudaSearchInterval(uint64_t lowerbound, uint64_t upperbound) 
{
	state = SIS_NEW;

	// Prepare interval storage
	set_extrema_safely(lowerbound, upperbound);
	cudaError_t err = cudaMalloc(&internal_storage, capacity * sizeof(uint64_t));
	if(err != cudaSuccess) {
		std::cout << "Couldn't actuate space memory on the Jeep U. Everything is fucked, Bubbs!" << std::endl;
	} else {
		initialize();
	}

	// Prepare device prime storage
	err = cudaMalloc(&device_primes, capacity * sizeof(uint64_t));
	if(err != cudaSuccess) {
		std::cout << "Couldn't actuate space memory on the Jeep U. Everything is fucked, Bubbs!" << std::endl;
	}
};

CudaSearchInterval::~CudaSearchInterval() {
	cudaFree(internal_storage);
	cudaFree(device_primes);
};

void CudaSearchInterval::repopulate(uint64_t lowerbound, uint64_t upperbound) {
	set_extrema_safely(lowerbound, upperbound);
	initialize();
};

uint64_t CudaSearchInterval::lowerbound() const {
	return lb;
};

uint64_t CudaSearchInterval::upperbound() const {
	return ub;
};

uint64_t CudaSearchInterval::smallest_multiple(uint64_t prime) {
	uint64_t residue = (lb % prime);
	if (residue == 0) {
		// lowerbound is a multiple of prime
		return lb;
	} else {
		// lowerbound - residue is a multiple of prime
		// but it lies outside the range, so increment by prime
		return lb - residue + prime;
	}	
};

uint64_t CudaSearchInterval::largest_multiple(uint64_t prime) {
	// If residue is 7, then upperbound is 7 greater
	// than largest multiple of prime in interval
	uint64_t residue = ub % prime;
	return ub - residue;
}
void CudaSearchInterval::mark_composite(uint64_t composite) {
	uint64_t index = composite - lb;
	if (index <= population) {
		internal_storage[index] = COMPOSITE;
	} else {
		std::cout << "Attempted to access composite outside of bounds" << std::endl;
	}
};

std::vector<uint64_t> CudaSearchInterval::get_primes() {
	std::vector<uint64_t> primes;
	for(uint64_t i = 0; i < population; i++) {
		uint64_t candidate = internal_storage[i];
		if(candidate != COMPOSITE) {
			primes.push_back(candidate);
		}
	}
	return primes;
};

void CudaSearchInterval::mark_multiples_of_prime(uint64_t prime) {
	// Determine smallest composite in range (start)
	uint64_t start = smallest_multiple(prime);

	// Determine largest composite in range (stop)
	uint64_t stop = largest_multiple(prime);

	// Mark all multiples of prime
	for(uint64_t composite = start; composite <= stop; composite += prime) {
		mark_composite(composite);
	}
};

std::ostream& operator<< (std::ostream& o, const CudaSearchInterval& si) {
		return o << "[" << si.lowerbound() << "," << si.upperbound() << "]";
};

