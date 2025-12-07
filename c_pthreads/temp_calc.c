#define _POSIX_C_SOURCE 200112L  // для использования функций POSIX стандарта 200112
#define _GNU_SOURCE         

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Конфигурация задачи
#define N_X       100000      // кол-во узлов по пространству
#define N_THREADS 1           // кол-во потоков для распараллеливания

#define LAMBDA_VAL 46.0       
#define RHO        7800.0    
#define C_HEAT     460.0     
#define L          0.1       
#define TA         300.0     
#define TB         500.0     
#define T0         20.0      
#define T_FINAL    0.01      

// массивы для температур на текущем и следующем временных слоях
double *T_current;
double *T_next;

pthread_barrier_t barrier_calc;  // для синхронизации после расчета
pthread_barrier_t barrier_swap;  // для синхронизации после обмена указателей

double a, h, tau, r;  
int N_t;            

// Структура данных для передачи параметров потоку
typedef struct {
    int thread_id;     
    int start_index;   
    int end_index;    
} ThreadData;

void* worker_function(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int start = data->start_index;
    int end = data->end_index;

    double r_local = r;  // локальная копия для оптимизации доступа

    for (int n = 0; n < N_t; n++) {
        double *loc_cur = T_current;
        double *loc_nxt = T_next;

        // расчет по явной схеме для выделенного диапазона узлов
        for (int i = start; i <= end; i++) {
            loc_nxt[i] = loc_cur[i] + r_local * (loc_cur[i+1] - 2.0 * loc_cur[i] + loc_cur[i-1]);
        }
      
        pthread_barrier_wait(&barrier_calc);
        pthread_barrier_wait(&barrier_swap);
    }
    return NULL;
}

int main() {
    struct timespec start_time, end_time;  // для измерения времени выполнения

    // расчет параметров численной схемы
    a = LAMBDA_VAL / (RHO * C_HEAT);     
    h = L / (N_X - 1);                    
    r = 0.49;                             
    tau = r * (h * h) / a;              
    N_t = (int)ceil(T_FINAL / tau);       

    printf("Потоков: %d, Шагов по времени: %d, Время моделирования: %.2f с\n", N_THREADS, N_t, T_FINAL);

    // выделение памяти для температурных массивов
    T_current = (double*)malloc(sizeof(double) * N_X);
    T_next = (double*)malloc(sizeof(double) * N_X);

    for (int i = 0; i < N_X; i++) T_current[i] = T0;  // начальная температура во всем стержне
    T_current[0] = TA;                                 // температура на левом конце
    T_current[N_X - 1] = TB;                           // температура на правом конце
    memcpy(T_next, T_current, sizeof(double) * N_X);   // копирование для первого шага

    pthread_t threads[N_THREADS];           // массив идентификаторов потоков
    ThreadData thread_data[N_THREADS];      // массив данных для потоков

    // инициализация барьеров
    pthread_barrier_init(&barrier_calc, NULL, N_THREADS + 1);
    pthread_barrier_init(&barrier_swap, NULL, N_THREADS + 1);

    // разделение вычислительной области между потоками 
    int inner_points = N_X - 2;                
    int points_per_thread = inner_points / N_THREADS;
    int remainder = inner_points % N_THREADS;   
    int current_start = 1;                      

    for (int i = 0; i < N_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].start_index = current_start;
        int chunk_size = points_per_thread + (i < remainder ? 1 : 0);  // распределение остатка
        thread_data[i].end_index = current_start + chunk_size - 1;
        current_start += chunk_size;            // сдвиг начала для следующего потока
    }


    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (int i = 0; i < N_THREADS; i++) {
        pthread_create(&threads[i], NULL, worker_function, &thread_data[i]);
    }

    for (int n = 0; n < N_t; n++) {
        pthread_barrier_wait(&barrier_calc);

        // обмен указателями на массивы
        double *tmp = T_current;
        T_current = T_next;
        T_next = tmp;

        // обновление ГУ
        T_current[0] = TA;
        T_current[N_X - 1] = TB;

        pthread_barrier_wait(&barrier_swap);
    }

    // Ожидание завершения всех потоков
    for (int i = 0; i < N_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    long long elapsed_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000LL +
                           (end_time.tv_nsec - start_time.tv_nsec);
    double elapsed_s = (double)elapsed_ns / 1000000000.0;

    printf("Время выполнения: %.4f с\n", elapsed_s);
    printf("Температура в центре стержня: %.2f K\n", T_current[N_X/2]);

    pthread_barrier_destroy(&barrier_calc);
    pthread_barrier_destroy(&barrier_swap);
    free(T_current);
    free(T_next);

    return 0;
}
