#!/usr/bin/python

buckets = [0,0,0,0,0,0,0,0]
def sort_prime_into_bucket(p):
	if p < 10:
		buckets[1] += 1
	elif p < 100:
		buckets[2] += 1
	elif p < 1000:
		buckets[3] += 1
	elif p < 10000:
		buckets[4] += 1
	elif p < 100000:
		buckets[4] += 1
	elif p < 1000000:
		buckets[5] += 1
	elif p < 10000000:
		buckets[6] += 1
	elif p < 100000000:
		buckets[7] += 1
	print buckets

pi = 0
def log_prime(p):
	global pi
	pi += 1
	print "Prime[%d] := %d" % (pi, p)

def sieve(primes, lower_bound, upper_bound):
	new_primes = []
	for c in xrange(lower_bound, upper_bound):
		c_might_be_prime = True
		for p in primes:
			if c % p == 0:
				c_might_be_prime = False
		if c_might_be_prime:
			sort_prime_into_bucket(c)
			log_prime(c)
			new_primes.append(c)
	return new_primes

def next_region(current_upper_bound):
	cub = current_upper_bound
	lb = cub + 1
	ub = lb * lb
	return (lb, ub)

def advance(primes):
	lb, ub = next_region(primes[-1])
	new_primes = sieve(primes, lb, ub)
	return primes + new_primes

if __name__ == "__main__":
	primes = [2,3,5,7]
	while (primes[-1] < 10000):
		primes = advance(primes)
