#include <iostream>
#include <vector>
#include <cstdint>
#include <string>

#define MAX_INTERVAL_SIZE 1024 * 1024 * 10
class ResizeableSieveInterval {
		uint64_t lb;
		uint64_t ub;
		std::vector<uint64_t> interval;
	public:
		SieveInterval(uint64_t lowerbound, uint64_t upperbound) 
		{
			interval.reserve(MAX_INTERVAL_SIZE);
			resize(lowerbound, upperbound);
		};
		void resize(uint64_t lowerbound, uint64_t upperbound) {
			lb = lowerbound;
			ub = upperbound;
			if (ub - lb > MAX_INTERVAL_SIZE) {
				ub = lb + MAX_INTERVAL_SIZE;
			}
			for(uint64_t i = lb; i <= ub; i++) {
				interval.push_back(i);
			}
		}
		uint64_t lowerbound() const {
			return lb;
		};
		uint64_t upperbound() const {
			return ub;
		};
		void mark_composite(uint64_t composite) {
			uint64_t index = composite - lb;
			interval.at(index) = 0;
		};
		void append_primes(std::vector<uint64_t> &primes) {
			
			for(std::vector<uint64_t>::iterator it = interval.begin(); it != interval.end(); ++it) {
				uint64_t p = *it;
				if (p != 0) {
					primes.push_back(p);
				}
			}
		};
		friend std::ostream& operator<< (std::ostream& o, const SieveInterval& si) {
				return o << "[" << si.lowerbound() << "," << si.upperbound() << "]";
		};
};

void mark_multiples_of_prime(uint64_t prime, SieveInterval& si) {
	// Determine smallest composite in range (start)
	uint64_t lowerbound = si.lowerbound();
	uint64_t lb_mod_p = (lowerbound % prime);
	uint64_t start = (lb_mod_p == 0) ? lowerbound : lowerbound - lb_mod_p + prime;

	// Determine largest composite in range (stop)
	uint64_t upperbound = si.upperbound();
	uint64_t ub_mod_p = (upperbound % prime);
	uint64_t stop = upperbound - ub_mod_p;

	for(uint64_t composite = start; composite <= stop; composite += prime) {
		try {
				si.mark_composite(composite);
		} catch (std::out_of_range e) {
			std::cout << "Could not mark composite " << composite << " in " << si << std::endl;
			exit(0);
		}
	}
}

void apply_sieve(SieveInterval& si, std::vector<uint64_t> primes) {
	for(std::vector<uint64_t>::iterator it = primes.begin(); it != primes.end(); ++it) {
		uint64_t p = *it;
		mark_multiples_of_prime(p, si);
	}
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
		uint64_t lowerbound = primes.back() + 1;
		uint64_t upperbound = primes.back() * primes.back();
		if(upperbound > MAX_PRIME) { upperbound = MAX_PRIME; }
		SieveInterval si(lowerbound, upperbound);
		std::cout << "Applying Sieve to " << si << " -- ";
		apply_sieve(si, primes);
		si.append_primes(primes);
		std::cout << primes.size() << " primes" << std::endl;
		perform_another_sieve = (si.upperbound() < MAX_PRIME);
	}

	return 0;
}
