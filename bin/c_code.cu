#include <assert.h>
#include <cuda_runtime.h>

	#include <stdio.h>
	#include <stdlib.h>
	#include <time.h>

	int main(int argc, char *argv[])
	{
	if(argc != 2)
		exit(2);
	int N = atoi(argv[1]);
	
	int first[N][N], second[N][N], multiply[N][N];

	for (int i = 0; i < N; i++){
	    for (int j = 0; j < N; j++){
		first[i][j] = rand()%10;
		second[i][j] = rand()%10;
		multiply[i][j] = 1;
	}
	}

		//clock_t begin = clock();
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

		for (int i = 0; i < N; i++){
		    for (int j = 0; j < N; j++){
			int sum = 0;
			for (int k = 0; k < N; k++)
				sum += first[i][k] * second[k][j];
			multiply[i][j] = sum;
		}
		}
	//clock_t end = clock();
	//double time_spent = (double)(end - begin) / (CLOCKS_PER_SEC / 1000);
	//fprintf(stdout, "%f", time_spent);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
fprintf(stdout, "%f", milliseconds);    

	
	return 0;
	}

