#define _POSIX_C_SOURCE 200112L
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h> // библиотека OpenMP

#define N_X       100000 // Количество узлов
#define N_THREADS 48     // Количество потоков для OpenMP

#define LAMBDA_VAL 46.0
#define RHO        7800.0
#define C_HEAT     460.0
#define L          0.1
#define TA         300.0 // Левая граница (T(0, t))
#define TB         500.0 // Правая граница (T(L, t))
#define T0         20.0  // Однородное начальное условие (T(x, 0))
#define T_FINAL    0.01  // Финальное время

double *T_current;
double *T_next;

double a, h, tau, r;
int N_t;

int main() {
    struct timespec start_time, end_time;

    
    a = LAMBDA_VAL / (RHO * C_HEAT); 
    h = L / (N_X - 1);

    r = 0.49;
    tau = r * (h * h) / a;
    N_t = (int)ceil(T_FINAL / tau);

    if (r > 0.500001) {
        fprintf(stderr, "Unstable r=%.4f\n", r);
        return 1;
    }

    // выделение памяти
    T_current = (double*)malloc(sizeof(double) * N_X);
    T_next = (double*)malloc(sizeof(double) * N_X);

    // инициализация начальных условий
    for (int i = 0; i < N_X; i++) {
        T_current[i] = T0;
    }

    // установка граничных условий на начальном слое
    T_current[0] = TA;
    T_current[N_X - 1] = TB;

    memcpy(T_next, T_current, sizeof(double) * N_X);
    omp_set_num_threads(N_THREADS);

    // ЗАМЕР ВРЕМЕНИ
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (int n = 0; n < N_t; n++) {
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < N_X - 1; i++) {
            // T_i^{n+1} = T_i^n + r * (T_{i+1}^n - 2*T_i^n + T_{i-1}^n)
            T_next[i] = T_current[i] + r * (T_current[i+1] - 2.0 * T_current[i] + T_current[i-1]);
        }

        double *tmp = T_current;
        T_current = T_next;
        T_next = tmp;
        T_current[0] = TA;
        T_current[N_X - 1] = TB;
    }

  
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    long long elapsed_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000LL +
                           (end_time.tv_nsec - start_time.tv_nsec);
    double elapsed_s = (double)elapsed_ns / 1000000000.0;

    printf("--- C (OpenMP) с однородным НУ (%d потоков) ---\n", N_THREADS);
    printf("Шагов N_t: %d\n", N_t);
    printf("Температура в центре T: %.2f °C\n", T_current[N_X/2]);
    printf("Время выполнения: %.4f с\n", elapsed_s);

    free(T_current);
    free(T_next);

    return 0;
}
