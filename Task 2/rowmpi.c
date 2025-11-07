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
    int rows_per_proc = Size / ProcNum;
    if (ProcRank == 0) {
        printf("=== Параллельное умножение матрицы на вектор ===\n");
        printf("Количество процессов: %d\n", ProcNum);
        printf("Размер матрицы: %dx%d\n", Size, Size);
        printf("Строк на процесс: %d\n", rows_per_proc);
    }

    double* local_matrix = (double*)malloc(rows_per_proc * Size * sizeof(double));
    double* vector = (double*)malloc(Size * sizeof(double));
    double* local_result = (double*)malloc(rows_per_proc * sizeof(double));
    double* global_result = NULL;

    if (ProcRank == 0) {
        global_result = (double*)malloc(Size * sizeof(double));
    }

    //=== Подготовка и распределение данных ===
    if (ProcRank == 0) {
        printf("Инициализация и распределение данных...\n");
        double* full_matrix = (double*)malloc(Size * Size * sizeof(double));
        srand(time(NULL));
        for (int i = 0; i < Size * Size; i++) {
            full_matrix[i] = (double)rand() / RAND_MAX * 10.0;
        }
        for (int i = 0; i < Size; i++) {
            vector[i] = (double)rand() / RAND_MAX * 10.0;
        }
        //Распределение матрицы
        MPI_Scatter(full_matrix, rows_per_proc * Size, MPI_DOUBLE,
                   local_matrix, rows_per_proc * Size, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        free(full_matrix);
        printf("Данные распределены\n");
    } else {
        //Процессы с рангом > 0 получают данные
        MPI_Scatter(NULL, rows_per_proc * Size, MPI_DOUBLE,
                   local_matrix, rows_per_proc * Size, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
    }
    MPI_Bcast(vector, Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (ProcRank == 0) {
        printf("Вектор разослан\n");
    }
    //Синхронизация
    MPI_Barrier(MPI_COMM_WORLD);
    if (ProcRank == 0) printf("Все процессы синхронизированы\n");

    //=== Последовательные вычисления (только процесс 0) ===
    double sequential_time = 0.0;
    if (ProcRank == 0) {
        printf("Измерение последовательного времени...\n");
        double* test_matrix = (double*)malloc(Size * Size * sizeof(double));
        double* test_vector = (double*)malloc(Size * sizeof(double));
        double* test_result = (double*)malloc(Size * sizeof(double));
        //Простая инициализация для теста
        for (int i = 0; i < Size * Size; i++) test_matrix[i] = 1.0;
        for (int i = 0; i < Size; i++) test_vector[i] = 1.0;
        double start = MPI_Wtime();
        for (int i = 0; i < Size; i++) {
            test_result[i] = 0.0;
            for (int j = 0; j < Size; j++) {
                test_result[i] += test_matrix[i * Size + j] * test_vector[j];
            }
        }
        double end = MPI_Wtime();
        sequential_time = end - start;
        printf("Последовательное время: %.6f сек\n", sequential_time);

        free(test_matrix);
        free(test_vector);
        free(test_result);
    }
    MPI_Bcast(&sequential_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //Синхронизация
    MPI_Barrier(MPI_COMM_WORLD);
    if (ProcRank == 0) printf("Начало параллельных вычислений...\n");

    //=== Параллельные вычисления ===
    double parallel_start = MPI_Wtime();
    for (int i = 0; i < rows_per_proc; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < Size; j++) {
            local_result[i] += local_matrix[i * Size + j] * vector[j];
        }
    }
    double parallel_end = MPI_Wtime();
    double parallel_computation_time = parallel_end - parallel_start;

    if (ProcRank == 0) {
        printf("Параллельные вычисления завершены\n");
    }

    //=== Сбор результатов ===
    double gather_start = MPI_Wtime();
    MPI_Gather(local_result, rows_per_proc, MPI_DOUBLE,
              global_result, rows_per_proc, MPI_DOUBLE,
              0, MPI_COMM_WORLD);
    double gather_end = MPI_Wtime();
    double gather_time = gather_end - gather_start;

    if (ProcRank == 0) {
        printf("Результаты собраны\n");
    }
    double max_comp_time;
    MPI_Reduce(&parallel_computation_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    //=== Анализ производительности ===
    if (ProcRank == 0) {
        printf("\n=== Результаты ===\n");
        printf("Последовательное время: %.6f сек\n", sequential_time);
        printf("Параллельное время вычислений: %.6f сек\n", max_comp_time);
        printf("Время сбора результатов: %.6f сек\n", gather_time);
        double speedup = sequential_time / max_comp_time;
        double efficiency = speedup / ProcNum;

        printf("\n=== Анализ производительсности ===\n");
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
    free(vector);
    free(local_result);
    if (ProcRank == 0) free(global_result);

    MPI_Finalize();
    return 0;
}
