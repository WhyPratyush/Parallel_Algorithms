//bench-flags: -march=native
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

static inline uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *state = x;
}

void generate_inputs(double *arr, size_t n, uint64_t seed) {
    uint64_t state = seed ? seed : 42ULL;
    for(size_t i = 0; i < n; i++) {
        arr[i] = (double)(xorshift64(&state) >> 11) * (1.0 / 9007199254740992.0);
    }
}


double sequential_sum(const double *arr, size_t n) {
    double sum = 0.0;
    for(size_t i = 0; i < n; i++) sum += arr[i];
    return sum;
}

int main(int argc, char* argv[]) {
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

    double *arr = malloc(sizeof(double) * n);
    if(!arr) {
        perror("Malloc failed");
        return EXIT_FAILURE;
    }

    generate_inputs(arr, n, 42);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    double sum = sequential_sum(arr, n);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_ms = ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9) * 1e3;
    printf("RESULT time_ms=%.6f result=%.17g\n", elapsed_ms, sum);

    free(arr);
    return EXIT_SUCCESS;
}