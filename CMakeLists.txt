cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)
project (Paratosthenes)
add_executable(Paratosthenes serial_sieve.cpp)
target_compile_features(Paratosthenes PRIVATE cxx_range_for)
