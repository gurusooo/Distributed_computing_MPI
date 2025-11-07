#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int ProcNum, ProcRank;
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    int Size = 5000;
    int cols_per_proc = Size / ProcNum;

    if (ProcRank == 0) {
        printf("=== Умножение матрицы на вектор - столбцы ===\n");
        printf("Количество процессов: %d\n", ProcNum);
        printf("Размер матрицы: %dx%d\n", Size, Size);
        printf("Столбцов на процесс: %d\n", cols_per_proc);
    }
    double* local_matrix = (double*)malloc(Size * cols_per_proc * sizeof(double));
    double* local_vector = (double*)malloc(cols_per_proc * sizeof(double));
    double* local_result = (double*)calloc(Size, sizeof(double));
    double* global_result = NULL;
    //Для проверки корректности на процессе 0
    double* full_matrix_check = NULL;
    double* full_vector_check = NULL;

    if (ProcRank == 0) {
        global_result = (double*)malloc(Size * sizeof(double));
        full_matrix_check = (double*)malloc(Size * Size * sizeof(double));
        full_vector_check = (double*)malloc(Size * sizeof(double));
    }

    //=== Подготовка и распределение данных ===
    if (ProcRank == 0) {
        printf("Инициализация и распределение данных...\n");
        srand(time(NULL));
        //Сохраняем данные для проверок
        for (int i = 0; i < Size * Size; i++) {
            full_matrix_check[i] = (double)rand() / RAND_MAX * 10.0;
        }
        for (int i = 0; i < Size; i++) {
            full_vector_check[i] = (double)rand() / RAND_MAX * 10.0;
        }

        //Распределение матрицы
        for (int proc = 0; proc < ProcNum; proc++) {
            for (int i = 0; i < Size; i++) {
                for (int j = 0; j < cols_per_proc; j++) {
                    int global_col = proc * cols_per_proc + j;
                    int local_index = i * cols_per_proc + j;
                    int global_index = i * Size + global_col;
                    if (proc == 0) {
                        local_matrix[local_index] = full_matrix_check[global_index];
                    } else {
                        MPI_Send(&full_matrix_check[global_index], 1, MPI_DOUBLE,
                                proc, 0, MPI_COMM_WORLD);
                    }
                }
            }
        }

        //Распределение вектора
        for (int proc = 0; proc < ProcNum; proc++) {
            for (int j = 0; j < cols_per_proc; j++) {
                int global_index = proc * cols_per_proc + j;
                if (proc == 0) {
                    local_vector[j] = full_vector_check[global_index];
                } else {
                    MPI_Send(&full_vector_check[global_index], 1, MPI_DOUBLE,
                            proc, 0, MPI_COMM_WORLD);
                }
            }
        }
        printf("Данные распределены\n");
    } else {
        // Процессы с рангом > 0 получают данные
        for (int i = 0; i < Size; i++) {
            for (int j = 0; j < cols_per_proc; j++) {
                MPI_Recv(&local_matrix[i * cols_per_proc + j], 1, MPI_DOUBLE,
                        0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (int j = 0; j < cols_per_proc; j++) {
            MPI_Recv(&local_vector[j], 1, MPI_DOUBLE,
                    0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //=== Точное измерение последовательного времени ===
    double sequential_time = 0.0;
    if (ProcRank == 0) {
        printf("Точное измерение последовательного времени...\n");
        double* test_result = (double*)malloc(Size * sizeof(double));
        for (int i = 0; i < Size; i++) {
            test_result[i] = 0.0;
            for (int j = 0; j < Size; j++) {
                test_result[i] += full_matrix_check[i * Size + j] * full_vector_check[j];
            }
        }
        //Повыс точность
        int num_runs = 3;
        double total_time = 0.0;
        for (int run = 0; run < num_runs; run++) {
            double start = MPI_Wtime();
            for (int i = 0; i < Size; i++) {
                test_result[i] = 0.0;
                for (int j = 0; j < Size; j++) {
                    test_result[i] += full_matrix_check[i * Size + j] * full_vector_check[j];
                }
            }
            double end = MPI_Wtime();
            total_time += (end - start);
        }
        sequential_time = total_time / num_runs;
        printf("Усредненное последовательное время (%d запусков): %.6f сек\n", num_runs, sequential_time);

        free(test_result);
    }

    MPI_Bcast(&sequential_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (ProcRank == 0) printf("Начало параллельных вычислений...\n");

    //=== Параллельные вычисления ===
    double parallel_start = MPI_Wtime();
    for (int i = 0; i < Size; i++) {
        for (int j = 0; j < cols_per_proc; j++) {
            local_result[i] += local_matrix[i * cols_per_proc + j] * local_vector[j];
        }
    }
    double parallel_end = MPI_Wtime();
    double parallel_computation_time = parallel_end - parallel_start;

    //=== Сбор результатов ===
    MPI_Reduce(local_result, global_result, Size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (ProcRank == 0) {
        printf("Результаты собраны\n");
    }

    //=== Проверка корректности и анализ ===
    if (ProcRank == 0) {
        printf("Проверка корректности результатов...\n");
        double* sequential_result = (double*)malloc(Size * sizeof(double));
        for (int i = 0; i < Size; i++) {
            sequential_result[i] = 0.0;
            for (int j = 0; j < Size; j++) {
                sequential_result[i] += full_matrix_check[i * Size + j] * full_vector_check[j];
            }
        }
        int correct = 1;
        double max_error = 0.0;
        for (int i = 0; i < Size; i++) {
            double error = fabs(global_result[i] - sequential_result[i]);
            if (error > max_error) {
                max_error = error;
            }
            if (error > 1e-6) {
                correct = 0;
                printf("Ошибка в строке %d: параллельный=%.6f, последовательный=%.6f, разница=%.6f\n",
                       i, global_result[i], sequential_result[i], error);
                if (i > 10) break;
            }
        }

        printf("Результаты %s (максимальная ошибка: %.6f)\n",
               correct ? "КОРРЕКТНЫ" : "СОДЕРЖАТ ОШИБКИ", max_error);

        free(sequential_result);
    }

    //=== Анализ производительности ===
    double max_comp_time;
    MPI_Reduce(&parallel_computation_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (ProcRank == 0) {
        printf("\n=== Результаты по столбцам ===\n");
        printf("Последовательное время: %.6f сек\n", sequential_time);
        printf("Параллельное время вычислений: %.6f сек\n", max_comp_time);
        double speedup = sequential_time / max_comp_time;
        double efficiency = speedup / ProcNum;
        printf("Ускорение (Sp): %.2f\n", speedup);
        printf("Эффективность (Ep): %.2f (%.1f%%)\n", efficiency, efficiency * 100);
        printf("Теоретическое ускорение: %.2f\n", (double)ProcNum);
        printf("\n=== Качество программы ===\n");
        if (efficiency > 1.0) {
            printf("Ошибка оценки эффективности\n");
        } else if (efficiency > 0.8) {
            printf("Отличная эффективность\n");
        } else if (efficiency > 0.6) {
            printf("Хорошая эффективность\n");
        } else {
            printf("Низкая эффективность\n");
        }
    }

    // Освобождение памяти
    free(local_matrix);
    free(local_vector);
    free(local_result);
    if (ProcRank == 0) {
        free(global_result);
        free(full_matrix_check);
        free(full_vector_check);
    }

    MPI_Finalize();
    return 0;
}
