/**
 * 实验五：Cache实验 - 矩阵乘法（优化算法）
 * 优化代码：i-k-j循环顺序
 * 空间局部性：b[]步长优化为1，所有数组访问步长均为1
 *
 * 优化原理：
 * 一般算法中遍历b[]的步长为size（列访问），空间局部性差
 * 优化算法将b[]的每次访问步长优化为1（行访问），提高空间局部性
 */
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    float *a, *b, *c, temp;
    long int i, j, k, size, m;
    struct timeval time1, time2;

    if(argc < 2) {
        printf("\n\tUsage: %s <Row of square matrix>\n", argv[0]);
        exit(-1);
    }

    size = atoi(argv[1]);
    m = size * size;

    // 分配内存
    a = (float*)malloc(sizeof(float) * m);
    b = (float*)malloc(sizeof(float) * m);
    c = (float*)malloc(sizeof(float) * m);

    // 初始化矩阵a和b
    for(i = 0; i < size; i++) {
        for(j = 0; j < size; j++) {
            a[i * size + j] = (float)(rand() % 1000 / 100.0);
            b[i * size + j] = (float)(rand() % 1000 / 100.0);
        }
    }

    // 清空结果矩阵c
    for(i = 0; i < size; i++) {
        for(j = 0; j < size; j++)
            c[i * size + j] = 0;
    }

    // 开始计时
    gettimeofday(&time1, NULL);

    // 优化后的矩阵乘法：i-k-j循环
    // 遍历a[]的每个元素，将每个元素的贡献累加到c[]中
    // temp = a[i][k]，在所有j上累加 temp * b[k][j] 到 c[i][j]
    for(i = 0; i < size; i++) {
        for(k = 0; k < size; k++) {
            temp = a[i * size + k];
            for(j = 0; j < size; j++)
                c[i * size + j] += temp * b[k * size + j];
        }
    }

    // 结束计时
    gettimeofday(&time2, NULL);

    // 计算时间差
    time2.tv_sec -= time1.tv_sec;
    time2.tv_usec -= time1.tv_usec;
    if(time2.tv_usec < 0L) {
        time2.tv_usec += 1000000L;
        time2.tv_sec -= 1;
    }

    printf("Matrix size: %ld x %ld\n", size, size);
    printf("Execution time = %ld.%06ld seconds\n", time2.tv_sec, time2.tv_usec);

    // 验证计算结果
    printf("Result check: c[0]=%.2f, c[%ld]=%.2f\n",
           c[0], m-1, c[m-1]);

    // 释放内存
    free(a);
    free(b);
    free(c);

    return(0);
}
