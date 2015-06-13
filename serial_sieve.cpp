#include <iostream>
#include <vector>
#include <cstdint>
#include <string>

std::vector<uint64_t> generate_range(uint64_t lowerbound, uint64_t upperbound) {
	uint64_t length = upperbound - lowerbound;
	std::vector<uint64_t> range;
	for(uint64_t i = lowerbound; i <= upperbound; i++) {
		range.push_back(i);
	}
	return range;
}

void mark_composites_in_range(uint64_t prime, std::vector<uint64_t> &range, uint64_t lowerbound, uint64_t upperbound) {
	// Determine smallest composite in range (start)
	uint64_t lb_mod_p = (lowerbound % prime);
	uint64_t start = (lb_mod_p == 0) ? lowerbound : lowerbound - lb_mod_p + prime;

	// Determine largest composite in range (stop)
	uint64_t ub_mod_p = (upperbound % prime);
	uint64_t stop = upperbound - ub_mod_p;

	// Adjust for appropriate index range
	start = start - lowerbound;
	stop = stop - lowerbound;
	for(uint64_t index = start; index <= stop; index += prime) {
		range.at(index) = 0;
	}
}

void apply_sieve(std::vector<uint64_t> &range, std::vector<uint64_t> primes) {
	uint64_t lowerbound = range.front();
	uint64_t upperbound = range.back();
	for(std::vector<uint64_t>::iterator it = primes.begin(); it != primes.end(); ++it) {
		uint64_t p = *it;
		std::cout << "Marking composites of " << p << std::endl;
		mark_composites_in_range(p, range, lowerbound, upperbound);
	}
}

void capture_primes_from_range(std::vector<uint64_t> &range, std::vector<uint64_t> &primes) {
	for(std::vector<uint64_t>::iterator it = range.begin(); it != range.end(); ++it) {
		if(*it != 0) {
			primes.push_back(*it);
		}
	}
}

void pruint64_t_list(std::vector<uint64_t> &list) {
	for(std::vector<uint64_t>::iterator it = list.begin(); it != list.end(); ++it) {
		std::cout << *it << ",";
	}
	std::cout << "\n";
}

int main(int argc, char** argv) {
	uint64_t MAX_PRIME = std::stoull(argv[1]);
	std::cout << "Finding all primes less than " << MAX_PRIME << "\n";
	
	// Initialize Primes
	std::vector<uint64_t> primes;
	primes.push_back(2);
	primes.push_back(3);
	primes.push_back(5);


	bool perform_another_sieve = (primes.back() < MAX_PRIME);
	while(perform_another_sieve) {
		uint64_t start = primes.back() + 1;
		uint64_t stop = primes.back() * primes.back();
		if (stop >= MAX_PRIME) {
			stop = MAX_PRIME;
		}
		std::vector<uint64_t> range = generate_range(start, stop);
		apply_sieve(range, primes);
		capture_primes_from_range(range, primes);
		std::cout << "There are " << primes.size() << " primes below " << stop << "\n";
		perform_another_sieve = (stop < MAX_PRIME);
	}

	return 0;
}
