enum SearchIntervalState { SIS_NEW, SIS_READY, SIS_ACTIVE, SIS_INCONSISTENT };
class CudaSearchInterval {
		uint64_t  lb;
		uint64_t  ub;
		uint64_t  population;
		uint64_t  capacity;
		uint64_t* internal_storage;
		SearchIntervalState state;
		void set_extrema_safely(uint64_t lowerbound, uint64_t upperbound);
	public:
		CudaSearchInterval(uint64_t lowerbound, uint64_t upperbound) ;
		~CudaSearchInterval();
		void repopulate(uint64_t lowerbound, uint64_t upperbound);
		void initialize();
		uint64_t lowerbound() const;
		uint64_t upperbound() const;
		uint64_t smallest_multiple(uint64_t prime);
		uint64_t largest_multiple(uint64_t prime); 
		void mark_composite(uint64_t composite);
		std::vector<uint64_t> get_primes();
		void mark_multiples_of_prime(uint64_t prime);
		void apply_sieve(std::vector<uint64_t> primes);
		friend std::ostream& operator<< (std::ostream& o, const CudaSearchInterval& si);
};
