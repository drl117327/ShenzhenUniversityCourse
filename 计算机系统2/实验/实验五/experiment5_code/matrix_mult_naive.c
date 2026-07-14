/**
 * 实验五：Cache实验 - 矩阵乘法（一般算法）
 * 代码A：i-k-j循环顺序（遍历结果矩阵每一行每一列）
 * 空间局部性：a[]步长为1（良好），b[]步长为size（较差）
 */
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    float *a, *b, *c;
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

    // 开始计时
    gettimeofday(&time1, NULL);

    // 一般矩阵乘法：i-j-k循环
    // 外层遍历行(i)，中层遍历列(j)，内层进行累加(k)
    for(i = 0; i < size; i++) {
        for(j = 0; j < size; j++) {
            c[i * size + j] = 0;
            for(k = 0; k < size; k++)
                c[i * size + j] += a[i * size + k] * b[k * size + j];
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

    // 验证计算结果（防止编译器优化掉计算）
    printf("Result check: c[0]=%.2f, c[%ld]=%.2f\n",
           c[0], m-1, c[m-1]);

    // 释放内存
    free(a);
    free(b);
    free(c);

    return(0);
}
