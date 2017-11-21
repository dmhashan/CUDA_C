cat > gcu.cu << EOF1

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#define BLOCK_SIZE $1

__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

__global__ void gpu_matrix_transpose(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols) 
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) 
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}

int main(int argc, char const *argv[])
{
    if(argc != 2)
	exit(2);
    int N = atoi(argv[1]);    

    /* Fixed seed for illustration */
    srand(3333);
    
    int *h_a, *h_b, *h_c;
    cudaMallocHost((void **) &h_a, sizeof(int)*N*N);
    cudaMallocHost((void **) &h_b, sizeof(int)*N*N);
    cudaMallocHost((void **) &h_c, sizeof(int)*N*N);
    
    // random initialize matrix A
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_a[i * N + j] = rand() % 10;
	  h_b[i * N + j] = rand() % 10;
        }
    }

    ///GPU
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int)*N*N);
    cudaMalloc((void **) &d_b, sizeof(int)*N*N);
    cudaMalloc((void **) &d_c, sizeof(int)*N*N);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*N*N, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
	    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N, N, N);    
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
fprintf(stdout, "%f", milliseconds);    

    cudaMemcpy(h_c, d_c, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    return 0;
}

EOF1

nvcc gcu.cu -o gcu
