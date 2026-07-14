/**
 * Memory Mountain 测量程序 (v5 final)
 * 使用 compiler barrier 防止优化，同时保留正常 cache 行为
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXSIZE     (64UL << 20)
#define MAXELEMS    (MAXSIZE / sizeof(long))
#define STRIDE_MAX  16

static long *data;
static volatile long global_sink;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* noinline + 返回有用值 */
__attribute__((noinline))
static long test(int elems, int stride) {
    long i;
    long sx2 = stride * 2, sx3 = stride * 3, sx4 = stride * 4;
    long acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    long length = elems;
    long limit = length - sx4;

    for (i = 0; i < limit; i += sx4) {
        acc0 += data[i];
        acc1 += data[i + stride];
        acc2 += data[i + sx2];
        acc3 += data[i + sx3];
    }
    for (; i < length; i += stride)
        acc0 += data[i];

    return acc0 + acc1 + acc2 + acc3;
}

static double measure_mb_per_sec(int elems, int stride) {
    long iterations;
    double best_time = 1e30;
    int repeats = 3;

    /* 估算迭代次数 */
    double t0 = now_sec();
    test(elems, stride);
    double t1 = now_sec();
    double single = t1 - t0;
    if (single < 1e-8) single = 1e-8;
    iterations = (long)(0.3 / single);
    if (iterations < 5) iterations = 5;
    if (iterations > 200000) iterations = 200000;

    for (int r = 0; r < repeats; r++) {
        long sum = 0;
        double t_start = now_sec();
        for (long k = 0; k < iterations; k++) {
            sum += test(elems, stride);
        }
        double t_end = now_sec();
        /* compiler barrier: 强制 sum 必须被计算 */
        __asm__ __volatile__("" : "+r"(sum) : : "memory");
        global_sink = sum;

        double elapsed = t_end - t_start;
        if (elapsed < best_time) best_time = elapsed;
    }

    long elems_per_call = elems / stride;
    if (elems_per_call < 1) elems_per_call = 1;
    double total_bytes = (double)elems_per_call * sizeof(long) * iterations;
    return (total_bytes / best_time) / 1e6;
}

int main(void) {
    fprintf(stderr, "Allocating %lu MB...\n", MAXSIZE >> 20);

    data = (long*)malloc(MAXSIZE);
    if (!data) { fprintf(stderr, "malloc failed!\n"); return 1; }
    for (unsigned long i = 0; i < MAXELEMS; i++)
        data[i] = (long)(i * 7919 + 1);

    fprintf(stderr, "Measuring (v5)...\n");

    printf("# Memory Mountain Data (v5)\n");
    printf("# size_bytes");
    for (int s = 1; s <= STRIDE_MAX; s++) printf(",S%d", s);
    printf("\n");
    fflush(stdout);

    unsigned long size = 16UL * 1024;
    while (size <= MAXSIZE) {
        int elems = (int)(size / sizeof(long));
        printf("%lu", size);
        fflush(stdout);

        for (int stride = 1; stride <= STRIDE_MAX; stride++) {
            double mbps = measure_mb_per_sec(elems, stride);
            printf(", %.1f", mbps);
            fflush(stdout);
        }
        printf("\n");

        size = size + (size >> 2);
        if (size < 16384) break;
    }

    fprintf(stderr, "Done! sink=%ld\n", global_sink);
    free(data);
    return 0;
}
