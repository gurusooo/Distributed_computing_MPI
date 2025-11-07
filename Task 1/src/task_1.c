//похоже я ошиблась с языком, сделала на си, та же логика, что на с++
#define _CRT_SECURE_NO_WARNINGS
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long total_points = 10000000;
    if (argc > 1) total_points = atoll(argv[1]);

    long long local_points = total_points / size;
    long long local_hits = 0;

    srand((unsigned int)(time(NULL)) + rank * 1000);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (long long i = 0; i < local_points; ++i) {
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y <= 1.0)
            local_hits++;
    }

    long long total_hits = 0;
    MPI_Reduce(&local_hits, &total_hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    double total_time = MPI_Wtime() - start_time;

    if (rank == 0) {
        double pi_final = 4.0 * (double)total_hits / total_points;

        char filename[50];
        sprintf(filename, "results_processes_%d.csv", size);

        FILE* result_file = fopen(filename, "w");
        if (result_file != NULL) {
            fprintf(result_file, "Processes,Time,Pi\n");
            fprintf(result_file, "%d,%.6f,%.10f\n", size, total_time, pi_final);
            fclose(result_file);

            printf("Processes: %d | Time: %.6fs | Pi: %.10f\n", size, total_time, pi_final);
            printf("Saved to: %s\n", filename);
        }
        else {
            printf("Error: could not open the file %s\n", filename);
        }
    }

    MPI_Finalize();
    return 0;
}
