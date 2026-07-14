/**
 * 实验五：矩阵乘法基准测试
 * 同时测试naive(i-j-k)和optimized(i-k-j)两种算法
 * 使用 clock_gettime(CLOCK_MONOTONIC) 获得可靠计时
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double get_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    if(argc < 2) {
        printf("Usage: %s <size>\n", argv[0]);
        return 1;
    }

    long size = atol(argv[1]);
    long m = size * size;
    double t_start, t_end;
    volatile double check = 0;  // prevent optimization

    float *a = (float*)malloc(sizeof(float) * m);
    float *b = (float*)malloc(sizeof(float) * m);
    float *c = (float*)malloc(sizeof(float) * m);

    // ---- Init ----
    for(long i = 0; i < m; i++) {
        a[i] = (float)(rand() % 1000 / 100.0);
        b[i] = (float)(rand() % 1000 / 100.0);
    }

    // ==== NAIVE: i-j-k ====
    for(long i = 0; i < size; i++)
        for(long j = 0; j < size; j++)
            c[i * size + j] = 0;

    t_start = get_sec();
    for(long i = 0; i < size; i++) {
        for(long j = 0; j < size; j++) {
            float sum = 0;
            for(long k = 0; k < size; k++)
                sum += a[i * size + k] * b[k * size + j];
            c[i * size + j] = sum;
            check += sum;
        }
    }
    t_end = get_sec();
    double t_naive = t_end - t_start;
    printf("NAIVE    | size=%5ld | time=%10.6f s | check=%.0f\n", size, t_naive, check);

    // ==== OPTIMIZED: i-k-j ====
    for(long i = 0; i < size; i++)
        for(long j = 0; j < size; j++)
            c[i * size + j] = 0;

    volatile double check2 = 0;
    t_start = get_sec();
    for(long i = 0; i < size; i++) {
        for(long k = 0; k < size; k++) {
            float temp = a[i * size + k];
            for(long j = 0; j < size; j++)
                c[i * size + j] += temp * b[k * size + j];
        }
    }
    t_end = get_sec();
    for(long j = 0; j < size; j++) check2 += c[j];  // prevent optimization
    double t_opt = t_end - t_start;
    printf("OPTIMIZED| size=%5ld | time=%10.6f s | check=%.0f\n", size, t_opt, check2);

    // Summary line for easy parsing
    double speedup = t_naive / t_opt;
    printf("RESULT   | %ld %.6f %.6f %.3f\n", size, t_naive, t_opt, speedup);

    free(a); free(b); free(c);
    return 0;
}
