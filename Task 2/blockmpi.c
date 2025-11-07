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
    int grid_rows, grid_cols;

    if (ProcNum == 1) {
        grid_rows = 1; grid_cols = 1;
    } else if (ProcNum == 2) {
        grid_rows = 2; grid_cols = 1;  //Линейная сетка 2x1
    } else if (ProcNum == 4) {
        grid_rows = 2; grid_cols = 2;  //Квадратная сетка 2x2
    } else {
        //Для других случаев - линейная сетка
        grid_rows = ProcNum;
        grid_cols = 1;
    }

    //Проверка валидности рангов
    int proc_row = ProcRank / grid_cols;
    int proc_col = ProcRank % grid_cols;
    int rows_per_proc = Size / grid_rows;
    int cols_per_proc = Size / grid_cols;

    if (ProcRank == 0) {
        printf("=== Умножение матрицы на вектор - БЛОЧНОЕ РАЗБИЕНИЕ ===\n");
        printf("Количество процессов: %d (сетка %dx%d)\n", ProcNum, grid_rows, grid_cols);
        printf("Размер матрицы: %dx%d\n", Size, Size);
        printf("Блок на процесс: %dx%d элементов\n", rows_per_proc, cols_per_proc);
        printf("Объем данных на процесс: %d элементов\n", rows_per_proc * cols_per_proc);
    }

    //Выделение памяти
    double* local_matrix = (double*)malloc(rows_per_proc * cols_per_proc * sizeof(double));
    double* local_vector_part = (double*)malloc(cols_per_proc * sizeof(double));
    double* local_result = (double*)calloc(rows_per_proc, sizeof(double));
    double* global_result = NULL;
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

        for (int i = 0; i < Size * Size; i++) {
            full_matrix_check[i] = (double)rand() / RAND_MAX * 10.0;
        }
        for (int i = 0; i < Size; i++) {
            full_vector_check[i] = (double)rand() / RAND_MAX * 10.0;
        }
        //Распределение матрицы по блокам по существующим процессам
        for (int proc = 0; proc < ProcNum; proc++) {
            int target_proc_row = proc / grid_cols;
            int target_proc_col = proc % grid_cols;

            if (proc < ProcNum) {
                for (int local_i = 0; local_i < rows_per_proc; local_i++) {
                    for (int local_j = 0; local_j < cols_per_proc; local_j++) {
                        int global_i = target_proc_row * rows_per_proc + local_i;
                        int global_j = target_proc_col * cols_per_proc + local_j;
                        int local_index = local_i * cols_per_proc + local_j;
                        int global_index = global_i * Size + global_j;
                        if (proc == 0) {
                            local_matrix[local_index] = full_matrix_check[global_index];
                        } else {
                            MPI_Send(&full_matrix_check[global_index], 1, MPI_DOUBLE,
                                    proc, 0, MPI_COMM_WORLD);
                        }
                    }
                }
            }
        }
        //Распределение вектора по существующим процессам
        for (int proc_col = 0; proc_col < grid_cols; proc_col++) {
            for (int local_j = 0; local_j < cols_per_proc; local_j++) {
                int global_j = proc_col * cols_per_proc + local_j;
                for (int proc_row = 0; proc_row < grid_rows; proc_row++) {
                    int target_proc = proc_row * grid_cols + proc_col;
                    if (target_proc < ProcNum) {
                        if (target_proc == 0) {
                            local_vector_part[local_j] = full_vector_check[global_j];
                        } else {
                            MPI_Send(&full_vector_check[global_j], 1, MPI_DOUBLE,
                                    target_proc, 0, MPI_COMM_WORLD);
                        }
                    }
                }
            }
        }
        printf("Данные распределены\n");
    } else {
        for (int i = 0; i < rows_per_proc; i++) {
            for (int j = 0; j < cols_per_proc; j++) {
                MPI_Recv(&local_matrix[i * cols_per_proc + j], 1, MPI_DOUBLE,
                        0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (int j = 0; j < cols_per_proc; j++) {
            MPI_Recv(&local_vector_part[j], 1, MPI_DOUBLE,
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
        //Многократные измерения
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

    for (int local_i = 0; local_i < rows_per_proc; local_i++) {
        for (int local_j = 0; local_j < cols_per_proc; local_j++) {
            local_result[local_i] += local_matrix[local_i * cols_per_proc + local_j] * local_vector_part[local_j];
        }
    }
    double parallel_end = MPI_Wtime();
    double parallel_computation_time = parallel_end - parallel_start;

    //=== Сбор результатов ===
    if (ProcRank == 0) {
        printf("Сбор результатов...\n");
    }

    if (grid_cols == 1) {
        //Gather для линейной сетки
        MPI_Gather(local_result, rows_per_proc, MPI_DOUBLE,
                  global_result, rows_per_proc, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);
    } else {
        //Для квадратной сетки (2x2, 4x4 и т.д.)
        if (proc_col == 0) {
            for (int source_col = 1; source_col < grid_cols; source_col++) {
                int source_proc = proc_row * grid_cols + source_col;
                if (source_proc < ProcNum) {
                    double* temp_result = (double*)malloc(rows_per_proc * sizeof(double));
                    MPI_Recv(temp_result, rows_per_proc, MPI_DOUBLE, source_proc, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (int i = 0; i < rows_per_proc; i++) {
                        local_result[i] += temp_result[i];
                    }
                    free(temp_result);
                }
            }
            if (proc_row == 0) {
                for (int i = 0; i < rows_per_proc; i++) {
                    global_result[i] = local_result[i];
                }
                //Получаем результаты от других строк
                for (int source_row = 1; source_row < grid_rows; source_row++) {
                    int source_proc = source_row * grid_cols;
                    if (source_proc < ProcNum) {
                        double* temp_result = (double*)malloc(rows_per_proc * sizeof(double));
                        MPI_Recv(temp_result, rows_per_proc, MPI_DOUBLE, source_proc, 0,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        for (int i = 0; i < rows_per_proc; i++) {
                            int global_i = source_row * rows_per_proc + i;
                            global_result[global_i] = temp_result[i];
                        }
                        free(temp_result);
                    }
                }
            } else if (proc_row * grid_cols < ProcNum) {
                MPI_Send(local_result, rows_per_proc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        } else if (ProcRank < ProcNum) {
            int target_proc = proc_row * grid_cols;
            if (target_proc < ProcNum) {
                MPI_Send(local_result, rows_per_proc, MPI_DOUBLE, target_proc, 0, MPI_COMM_WORLD);
            }
        }
    }

    if (ProcRank == 0) {
        printf("Результаты собраны\n");
    }

    //=== Проверка корректности ===
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
            if (error > max_error) max_error = error;
            if (error > 1e-6) correct = 0;
        }
        printf("Результаты %s (максимальная ошибка: %.10f)\n",
               correct ? "КОРРЕКТНЫ" : "СОДЕРЖАТ ОШИБКИ", max_error);
        free(sequential_result);
    }

    //=== Анализ производительности ===
    double max_comp_time;
    MPI_Reduce(&parallel_computation_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (ProcRank == 0) {
        printf("\n=== Результаты по блокам ===\n");
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

    free(local_matrix);
    free(local_vector_part);
    free(local_result);
    if (ProcRank == 0) {
        free(global_result);
        free(full_matrix_check);
        free(full_vector_check);
    }

    MPI_Finalize();
    return 0;
}
