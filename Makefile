serial_sieve.exe: serial_sieve.cpp
	clang++ -std=c++11 serial_sieve.cpp -o serial_sieve.exe
	time ./serial_sieve.exe 

openmp_sieve.exe: openmp_sieve.cpp
	clang++ -std=c++11 -Xclang -fopenmp openmp_sieve.cpp -o openmp_sieve.exe
	OMP_NUM_THREADS=4 time ./openmp_sieve.exe


clean:
	rm -f *.exe
	rm -f *.o
