#include <mpi.h>
#include <iostream>
#include <fstream>
#include <random>
#include <string>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long total_points = 10000000;
    if (argc > 1) total_points = std::atoll(argv[1]);

    long long local_points = total_points / size;
    long long local_hits = 0;

    std::mt19937 rng(rank + time(nullptr));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (long long i = 0; i < local_points; ++i) {
        double x = dist(rng);
        double y = dist(rng);
        if (x * x + y * y <= 1.0)
            local_hits++;
    }

    long long total_hits = 0;
    MPI_Reduce(&local_hits, &total_hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    double total_time = MPI_Wtime() - start_time;

    if (rank == 0) {
        double pi_final = 4.0 * (double)total_hits / total_points;

        std::string filename = "results_processes_" + std::to_string(size) + ".csv";
        std::ofstream result_file(filename);

        result_file << "Processes,Time,Pi\n";
        result_file << size << "," << total_time << "," << pi_final << "\n";
        result_file.close();

        std::cout << "Processes: " << size << " | Time: " << total_time << "s | Pi: " << pi_final << std::endl;
        std::cout << "Saved to: " << filename << std::endl;
    }

    MPI_Finalize();
    return 0;
}
