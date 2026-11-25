#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define BLOCK 256

#define CUDA_CHECK(call) do {                                        \
    cudaError_t e = (call);                                          \
    if (e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",                  \
                cudaGetErrorString(e), __FILE__, __LINE__);          \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while (0)

static inline uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *state = x;
}

static void generate_inputs(double *arr, size_t n, uint64_t seed) {
    uint64_t state = seed ? seed : 42ULL;
    for (size_t i = 0; i < n; i++) {
        arr[i] = (double)(xorshift64(&state) >> 11) * (1.0 / 9007199254740992.0);
    }
}

__global__ void Cuda_sum_naive(const double *d_in, double *d_out, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(d_out, d_in[idx]);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <num_elements>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *endptr;
    long long parsed_n = strtoll(argv[1], &endptr, 10);
    if (*endptr != '\0' || parsed_n <= 0) {
        fprintf(stderr, "Invalid element count: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    size_t n = (size_t)parsed_n;
    size_t bytes = sizeof(double) * n;

    double *h_in = (double *)malloc(bytes);
    if (!h_in) {
        perror("Malloc failed");
        return EXIT_FAILURE;
    }
    generate_inputs(h_in, n, 42);

    CUDA_CHECK(cudaFree(0));

    double *d_in = NULL, *d_out = NULL;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));

    unsigned int blocksPerGrid = (unsigned int)((n + BLOCK - 1) / BLOCK);

    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));
    Cuda_sum_naive<<<blocksPerGrid, BLOCK>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));

    CUDA_CHECK(cudaEventRecord(start));
    Cuda_sum_naive<<<blocksPerGrid, BLOCK>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    double h_out = 0.0;
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));

    printf("RESULT time_ms=%.6f h2d_ms=%.6f d2h_ms=%.6f result=%.17g\n",
           kernel_ms, h2d_ms, d2h_ms, h_out);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);

    return EXIT_SUCCESS;
}