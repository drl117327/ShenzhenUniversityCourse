/**
 * TLB (Translation Lookaside Buffer) 大小测量
 * 原理：以 PAGE_SIZE 为步长访问数组元素（每个访问命中不同页面），
 *       逐渐增加访问的页面数，当超过 TLB 容量时吞吐量下降。
 *
 * i9-14900HX TLB 参数（Intel 手册）：
 *   L1 dTLB: 96 entries (4KB pages) → covers 384KB
 *   L2 STLB: 2048 entries (4KB pages) → covers 8MB
 *
 * 编译: gcc -O2 wsl_tlb.c -o tlb -lm
 * 运行: ./tlb > tlb_data.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PAGE_SIZE      4096UL
#define MAX_PAGES      65536UL    /* 最大 65536 页 = 256MB */
#define MAX_SIZE       (MAX_PAGES * PAGE_SIZE)
#define STRIDE_PAGES   1          /* 每次跳过1个页面 */
#define REPEATS        3
#define ACCESSES_PER_TEST 100000

static char *data;
static volatile long sink;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* 核心：以 PAGE_SIZE 为步长遍历 num_pages 个页面 */
__attribute__((noinline))
static void access_pages(int num_pages, int total_accesses) {
    long sum = 0;
    int page_stride = STRIDE_PAGES;
    int max_idx = num_pages * (PAGE_SIZE / sizeof(long));

    for (int n = 0; n < total_accesses; n++) {
        /* 以 PAGE_SIZE 步长访问，每次命中不同页面 */
        for (int i = 0; i < max_idx; i += page_stride * (PAGE_SIZE / sizeof(long))) {
            sum += ((long*)data)[i];
        }
    }
    sink = sum;
}

static double measure_ns_per_access(int num_pages) {
    int total = ACCESSES_PER_TEST;
    double best_time = 1e30;

    /* Warmup */
    access_pages(num_pages, total / 10);

    for (int r = 0; r < REPEATS; r++) {
        double t0 = now_sec();
        access_pages(num_pages, total);
        double t1 = now_sec();

        int total_elem_accesses = total * num_pages;
        double ns_per = (t1 - t0) * 1e9 / total_elem_accesses;

        if (ns_per < best_time) best_time = ns_per;
    }
    return best_time;
}

int main(void) {
    fprintf(stderr, "Allocating %lu MB for TLB test...\n", MAX_SIZE >> 20);

    data = (char*)malloc(MAX_SIZE);
    if (!data) { fprintf(stderr, "malloc failed!\n"); return 1; }

    /* 初始化：每个页面写一次以确保页表已建立 */
    for (unsigned long i = 0; i < MAX_PAGES; i++) {
        data[i * PAGE_SIZE] = (char)(i & 0xFF);
    }

    fprintf(stderr, "Measuring TLB (this takes a few minutes)...\n");

    printf("# TLB Measurement Data\n");
    printf("# Columns: num_pages, size_kb, ns_per_access\n");
    printf("# L1 dTLB expected at ~96 pages (384KB), L2 STLB at ~2048 pages (8MB)\n");
    printf("# num_pages, size_kb, ns_per_access\n");
    fflush(stdout);

    /* 从 8 页开始，逐步增加到最大 */
    for (int num_pages = 8; num_pages <= (int)MAX_PAGES; num_pages = (int)(num_pages * 1.15 + 1)) {
        double ns = measure_ns_per_access(num_pages);
        printf("%d, %lu, %.2f\n", num_pages, (unsigned long)num_pages * PAGE_SIZE / 1024, ns);
        fflush(stdout);

        if (num_pages >= MAX_PAGES / 2) break;
    }

    fprintf(stderr, "Done! sink=%ld\n", sink);
    free(data);
    return 0;
}
