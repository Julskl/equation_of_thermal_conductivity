#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const double L = 0.1;          
const double Ta = 300.0;      
const double Tb = 500.0;      
const double T0 = 20.0;        
const double T_final = 0.01; 
const int N_x = 100000;       

const double lambda_val = 46.0;  
const double rho = 7800.0;      
const double c = 460.0;         

const double alpha = lambda_val / (rho * c);  
const double h = L / (N_x - 1);              
const double r = 0.49;                      
// const double r = 0.025;

// функция обмена граничными значениями с соседними процессами
void exchange_boundaries(double *T, int L_local, int rank, int size) {
    int left_neighbor = rank - 1;
    int right_neighbor = rank + 1;

    // обмен с правым соседом
    if (rank < size - 1) {
        // отправляем T[L_local], получаем в T[L_local+1]
        MPI_Sendrecv(&T[L_local], 1, MPI_DOUBLE, right_neighbor, 0,
                     &T[L_local+1], 1, MPI_DOUBLE, right_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // обмен с левым соседом (посылаем первый внутренний узел, получаем ghost cell)
    if (rank > 0) {
        // отправляем T[1] соседу слева, получаем в T[0]
        MPI_Sendrecv(&T[1], 1, MPI_DOUBLE, left_neighbor, 1,
                     &T[0], 1, MPI_DOUBLE, left_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // номер текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size);       // общее количество процессов

    double tau = r * (h * h) / alpha;           
    int N_t = (int)ceil(T_final / tau);        
  
    if (r > 0.500001) {
        if (rank == 0) {
            fprintf(stderr, "\n Ошибка: схема неустойчива при r = %.4f \n", r);
        }
        MPI_Finalize();
        return 1;
    }

    int global_inner_points = N_x - 2;          
    int L_local = global_inner_points / size;   // базовое количество точек на процесс
    int remainder = global_inner_points % size; // остаток для равномерного распределения
    
    if (rank < remainder) L_local++;            
  
    int N_local = L_local + 2;                  
    double *T_curr = (double*)malloc(N_local * sizeof(double));  
    double *T_next = (double*)malloc(N_local * sizeof(double));  

    for (int i = 0; i < L_local; i++) {
        T_curr[i + 1] = T0;  
    }
  
    if (rank > 0) T_curr[0] = T0;
    if (rank < size - 1) T_curr[L_local + 1] = T0;

    MPI_Barrier(MPI_COMM_WORLD);           // синхронизация всех процессов
    double start_time = MPI_Wtime();       


    for (int n = 0; n < N_t; n++) {
        exchange_boundaries(T_curr, L_local, rank, size);
        
        // расчет по явной разностной схеме
        for (int i = 1; i <= L_local; i++) {
            T_next[i] = T_curr[i] + r * (T_curr[i+1] - 2 * T_curr[i] + T_curr[i-1]);
        }
        
        // применение ГУ
        if (rank == 0) {
            T_next[1] = Ta;  
        }
        if (rank == size - 1) {
            T_next[L_local] = Tb;  
        }
        
        // обмен указателей для перехода к следующему временному слою
        double *temp = T_curr;
        T_curr = T_next;
        T_next = temp;
    }

    MPI_Barrier(MPI_COMM_WORLD);         
    double end_time = MPI_Wtime();       

    if (rank == 0) {
        double *T_global = (double*)malloc(N_x * sizeof(double)); 
        T_global[0] = Ta;                 
        T_global[N_x - 1] = Tb;           

        // подг параметров для MPI_Gatherv
        int *recvcounts = (int*)malloc(size * sizeof(int)); 
        int *displs = (int*)malloc(size * sizeof(int));      
        int current_displ = 1;  
        
        for (int i = 0; i < size; i++) {
            int local_len = global_inner_points / size;
            if (i < remainder) local_len++;
            recvcounts[i] = local_len;
            displs[i] = current_displ;
            current_displ += local_len;
        }

        // сбор данных со всех процессов
        MPI_Gatherv(&T_curr[1], L_local, MPI_DOUBLE,
                    T_global, recvcounts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        printf("--- C (MPI) с начальным условием T0 ---\n");
        printf("Количество временных шагов N_t: %d\n", N_t);
        printf("Температура в центре (x = %.3f м): %.2f °C\n", L / 2.0, T_global[N_x / 2]);
        printf("Время выполнения: %.4f с\n", end_time - start_time);

        free(T_global);
        free(recvcounts);
        free(displs);
    } else {
        // все остальные процессы только отправляют данные
        MPI_Gatherv(&T_curr[1], L_local, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // освобождение памяти
    free(T_curr);
    free(T_next);

    MPI_Finalize();
    return 0;
}
