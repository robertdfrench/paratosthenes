#include <iostream>
#include <vector>

std::vector<int> generate_range(int lowerbound, int upperbound) {
	int length = upperbound - lowerbound;
	std::vector<int> range;
	for(int i = lowerbound; i <= upperbound; i++) {
		range.push_back(i);
	}
	return range;
}

void mark_composites_in_range(int prime, std::vector<int> &range, int lowerbound, int upperbound) {
	// Determine smallest composite in range (start)
	int lb_mod_p = (lowerbound % prime);
	int start = (lb_mod_p == 0) ? lowerbound : lowerbound - lb_mod_p + prime;

	// Determine largest composite in range (stop)
	int ub_mod_p = (upperbound % prime);
	int stop = upperbound - ub_mod_p;

	// Adjust for appropriate index range
	start = start - lowerbound;
	stop = stop - lowerbound;
	for(int index = start; index <= stop; index += prime) {
		std::cout << "\tMarking " << index + lowerbound << " as 0 at position " << index << "\n";
		range.at(index) = 0;
	}
}

void apply_sieve(std::vector<int> &range, std::vector<int> primes) {
	int lowerbound = range.front();
	int upperbound = range.back();
	for(std::vector<int>::iterator it = primes.begin(); it != primes.end(); ++it) {
		int p = *it;
		std::cout << "Attempting to sieve for prime " << p << "\n";
		mark_composites_in_range(p, range, lowerbound, upperbound);
	}
}

void capture_primes_from_range(std::vector<int> &range, std::vector<int> &primes) {
	for(std::vector<int>::iterator it = range.begin(); it != range.end(); ++it) {
		if(*it != 0) {
			primes.push_back(*it);
		}
	}
}

void print_list(std::vector<int> &list) {
	for(std::vector<int>::iterator it = list.begin(); it != list.end(); ++it) {
		std::cout << *it << ",";
	}
	std::cout << "\n";
}

#define MAX_PRIME 5000
int main(int argc, char** argv) {
	std::cout << "Finding all primes less than " << MAX_PRIME << "\n";
	
	// Initialize Primes
	std::vector<int> primes;
	primes.push_back(2);
	primes.push_back(3);
	primes.push_back(5);


	bool perform_another_sieve = (primes.back() < MAX_PRIME);
	while(perform_another_sieve) {
		int start = primes.back() + 1;
		int stop = primes.back() * primes.back();
		if (stop >= MAX_PRIME) {
			stop = MAX_PRIME;
		}
		std::vector<int> range = generate_range(start, stop);
		apply_sieve(range, primes);
		capture_primes_from_range(range, primes);
		print_list(primes);
		perform_another_sieve = (stop < MAX_PRIME);
	}

	return 0;
}
