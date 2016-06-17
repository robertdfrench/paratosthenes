#!/usr/bin/python

if __name__ == "__main__":
	primes = [2,3,5]
	new_primes = []
	for i in range(6,25):
		i_might_be_prime = True
		for p in primes:
			if i % p == 0:
				i_might_be_prime = False
		if i_might_be_prime:
			new_primes.append(i)
	print new_primes
