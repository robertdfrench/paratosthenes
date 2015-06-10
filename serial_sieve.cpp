#include <iostream>
#include <vector>

std::vector<int> generate_range(int lowerbound, int upperbound) {
	int length = upperbound - lowerbound;
	std::vector<int> range;
	for(int i = lowerbound + 1; i <= upperbound; i++) {
		range << i;
	}
	return range;
}

void apply_sieve(std::vector<int> range, std::vector<int> primes) {
	int lowerbound = range.front();
	int upperbound = range.back();
	for(std::vector<int>::iterator it = primes.begin(); it != primes.end(); ++it) {
		int p = *it;
		int composite = lowerbound / p + p;
		while(composite <= upperbound) {
			int index = composite - lowerbound;
			range.at(index) = 0;
		}
	}
}

int main(int argc, char** argv) {
	std::vector<int> primes;
	primes << 2;
	primes << 3;
	primes << 5;
	std::vector range = generate_range(6, 25);
	apply_sieve(range, primes);

	for(std::vector<int>::iterator it = range.begin(); it != range.end(); ++it) {
		std::cout << *it << ",";
	}
	std::cout << "\n";
	return 0;
}
