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

int main(int argc, char** argv) {
	std::vector<int> primes;
	primes.push_back(2);
	primes.push_back(3);
	primes.push_back(5);
	std::vector<int> range = generate_range(6, 25);
	apply_sieve(range, primes);

	for(std::vector<int>::iterator it = range.begin(); it != range.end(); ++it) {
		std::cout << *it << ",";
	}
	std::cout << "\n";
	return 0;
}
