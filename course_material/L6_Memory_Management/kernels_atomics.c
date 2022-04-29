// Kernel to test atomics
__kernel void atomic_test (__global unsigned int* T) {
    
    // Increment T atomically
    atomic_add(T, 1);
}