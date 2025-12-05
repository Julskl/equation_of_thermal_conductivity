#define _POSIX_C_SOURCE 200112L // стандарт POSIX для thread barriers
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define N_X       100000  // кол-во пространственных узлов
#define N_THREADS 19     // кол-во потоков для параллелизации


// физические параметры
#define LAMBDA_VAL 46.0
#define RHO        7800.0
#define C_HEAT     460.0
#define L          0.1
#define TA         300.0
#define TB         500.0
#define T0         20.0
#define T_FINAL    0.01

double *T_current; // текущий слой
double *T_next;   // следующий слой
pthread_barrier_t barrier;

double a, h, tau, r;
int N_t;

typedef struct {                  // структура данных для потоков
    int thread_id;
    int start_index;
    int end_index;
    char padding[64 - 3*sizeof(int)];
} ThreadData;


// функция выполняемая каждым рабочим потоком
void* worker_function(void* arg) {              
    ThreadData* data = (ThreadData*)arg;
    int start = data->start_index;
    int end = data->end_index;
    double *loc_cur = T_current;
    double *loc_nxt = T_next;

    for (int n = 0; n < N_t; n++) {
       // вычисление температуры для своего участка стержня
        for (int i = start; i <= end; i++) {
            loc_nxt[i] = loc_cur[i] + r * (loc_cur[i+1] - 2.0 * loc_cur[i] + loc_cur[i-1]);
        }
        pthread_barrier_wait(&barrier);   // обмен указателями делает только поток
        // меняем местами указатели
        double *tmp = loc_cur;
        loc_cur = loc_nxt;
        loc_nxt = tmp;
  
         // синхронизация перед началом следующего временного шага
        pthread_barrier_wait(&barrier);
    }
    return NULL;
}

int main() {
    struct timespec start_time, end_time;

    a = LAMBDA_VAL / (RHO * C_HEAT);
    h = L / (N_X - 1);
    tau = 0.49 * (h * h) / a;
    r = a * tau / (h * h);           // параметр устойчивости схемы
    N_t = (int)ceil(T_FINAL / tau);  // кол-во временных шагов


    if (r > 0.50000) {
        fprintf(stderr, "Unstable r=%.4f\n", r);
        return 1;
    }
  
    // выделение памяти для массивов температур
    T_current = (double*)malloc(sizeof(double) * N_X);
    T_next = (double*)malloc(sizeof(double) * N_X);

    // инициализация
    for (int i = 0; i < N_X; i++) T_current[i] = T0;
    T_current[0] = TA;
    T_current[N_X - 1] = TB;
    memcpy(T_next, T_current, sizeof(double) * N_X);

    pthread_t threads[N_THREADS];
    ThreadData thread_data[N_THREADS];

    pthread_barrier_init(&barrier, NULL, N_THREADS);

    // распределение работы между потоками (inner_points - внутренние точки)
    int inner_points = N_X - 2;
    int points_per_thread = inner_points / N_THREADS;
    int remainder = inner_points % N_THREADS;
    int current_start = 1;

    for (int i = 0; i < N_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].start_index = current_start;
        int chunk_size = points_per_thread + (i < remainder ? 1 : 0);
        thread_data[i].end_index = current_start + chunk_size - 1;
        current_start += chunk_size;
    }

    // ЗАМЕР ВРЕМЕНИ
    clock_gettime(CLOCK_MONOTONIC, &start_time);
  
    // cоздание и запуск потоков
    for (int i = 0; i < N_THREADS; i++) {
        pthread_create(&threads[i], NULL, worker_function, &thread_data[i]);
    }

    // ожидание завершения всех потоков
    for (int i = 0; i < N_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    long long elapsed_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000LL +
                           (end_time.tv_nsec - start_time.tv_nsec);
    double elapsed_s = (double)elapsed_ns / 1000000000.0;

    printf("time: %.4f s\n", elapsed_s);
    printf("centre T: %.2f\n", T_current[N_X/2]);

    pthread_barrier_destroy(&barrier);
    free(T_current);
    free(T_next);

    return 0;
}
