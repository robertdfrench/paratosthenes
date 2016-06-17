test: mpi_sieve.c mpi_sieve.h
	cc mpi_sieve.c -o paratosthenes.exe
	cp paratosthenes.exe $(MEMBERWORK)/stf007
	cd $(MEMBERWORK)/stf007 && aprun -n16 ./paratosthenes.exe
