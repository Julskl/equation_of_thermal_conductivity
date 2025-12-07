from mpi4py import MPI
import numpy as np
import math
import sys

L = 0.1          
Ta = 300.0       
Tb = 500.0     
T0 = 20.0       
T_final = 0.01   
N_x = 100000     

lambda_val = 46.0  
rho = 7800.0      
c = 460.0       

# расчет параметров численной схемы
alpha = lambda_val / (rho * c) 
h = L / (N_x - 1)              
r = 0.49                    

def exchange_boundaries(T, comm, rank, size):
    """Обмен ghost cells с соседними процессами"""
  
    # обмен с правым соседом
    if rank < size - 1:
        comm.Sendrecv(sendbuf=T[-2:-1], dest=rank+1, sendtag=0,
                      recvbuf=T[-1:], source=rank+1, recvtag=1)
    # обмен с левым соседом
    if rank > 0:
        comm.Sendrecv(sendbuf=T[1:2], dest=rank-1, sendtag=1,
                      recvbuf=T[0:1], source=rank-1, recvtag=0)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

  tau = r * (h * h) / alpha
    N_t = int(math.ceil(T_final / tau))

    if r > 0.500001:
        if rank == 0:
            sys.stderr.write(f"Схема неустойчива. r = {r:.4f}\n")
        return

    global_inner_points = N_x - 2          # внутренние точки
    L_local = global_inner_points // size  # базовое количество точек
    remainder = global_inner_points % size  # остаток
    
    if rank < remainder:  
        L_local += 1

    N_local = L_local + 2 
    T_curr = np.zeros(N_local, dtype=np.float64)
    T_next = np.zeros(N_local, dtype=np.float64)

    # инициализация начальных условий
    T_curr[1:L_local+1] = T0 

    if rank > 0: T_curr[0] = T0
    if rank < size - 1: T_curr[-1] = T0

    comm.Barrier()
    start_time = MPI.Wtime()

    # основной вычислительный цикл
    for n in range(N_t):
        exchange_boundaries(T_curr, comm, rank, size)  # обмен границами
   
        T_next[1:-1] = T_curr[1:-1] + r * (T_curr[2:] - 2*T_curr[1:-1] + T_curr[:-2])
        
        if rank == 0: T_next[1] = Ta
        if rank == size - 1: T_next[L_local] = Tb
        
        T_curr, T_next = T_next, T_curr 

    comm.Barrier()
    end_time = MPI.Wtime()

    send_buffer = T_curr[1:L_local+1].copy()  

    if rank == 0:
        T_global = np.zeros(N_x, dtype=np.float64)
        T_global[0] = Ta
        T_global[-1] = Tb

        counts = []
        displs = []
        current_displ = 1
        
        for i in range(size):
            l_len = global_inner_points // size
            if i < remainder: l_len += 1
            counts.append(l_len)
            displs.append(current_displ)
            current_displ += l_len
        
        counts = np.array(counts, dtype=np.int32)
        displs = np.array(displs, dtype=np.int32)

        comm.Gatherv(sendbuf=send_buffer,
                     recvbuf=(T_global, (counts, displs), MPI.DOUBLE),
                     root=0)

        print(f"Время: {end_time - start_time:.4f} s")
        print(f"Температура в центре T: {T_global[N_x // 2]:.2f}")
    else:
        comm.Gatherv(sendbuf=send_buffer, recvbuf=None, root=0)

if __name__ == "__main__":
    main()
