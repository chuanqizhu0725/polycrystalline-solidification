mpicc main.c -o main-mpi && rm -f *.vtk && mpirun -n 6 ./main-mpi