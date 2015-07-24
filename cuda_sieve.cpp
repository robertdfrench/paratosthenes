#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include "cuda_search_interval.h"

int main(int argc, char** argv) {
	uint64_t MAX_PRIME = std::stoull(argv[1]);
	std::cout << "Finding all primes less than " << MAX_PRIME << "\n";
	
	// Initialize Primes
	std::vector<uint64_t> primes;
	primes.push_back(2);
	primes.push_back(3);
	primes.push_back(5);

	CudaSearchInterval si(2, 5);
	bool perform_another_sieve = (primes.back() < MAX_PRIME);
	while(perform_another_sieve) {
		uint64_t lowerbound = primes.back() + 1;
		uint64_t upperbound = primes.back() * primes.back();
		if(upperbound > MAX_PRIME) { upperbound = MAX_PRIME; }
		si.repopulate(lowerbound, upperbound);
		std::cout << "Applying Sieve to " << si << " -- ";
		try {
			si.apply_sieve(primes, primes);
			std::cout << primes.size() << " primes" << std::endl;
			perform_another_sieve = false;
		} catch (std::exception& e) {
			std::cout << "Fucking delice error bubs" << std::endl;
			perform_another_sieve = false;
		}
	}

	return 0;
}
