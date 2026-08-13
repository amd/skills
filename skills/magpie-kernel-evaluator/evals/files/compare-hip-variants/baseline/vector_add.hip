// Simple HIP vector_add kernel for Magpie correctness testing.
// Compile: hipcc -g -O2 -o vector_add vector_add.hip
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(vector_add, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_a, d_b, d_c, N);

    hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at %d: got %f, expected %f\n",
                    i, h_c[i], expected);
            errors++;
        }
    }

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    if (errors == 0) {
        printf("PASSED: All %d elements correct.\n", N);
        return 0;
    } else {
        printf("FAILED: %d errors.\n", errors);
        return 1;
    }
}
