#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// Constants for WMMA
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

// Warps per block in M and N dimensions
#define WARPS_PER_BLOCK_M 2
#define WARPS_PER_BLOCK_N 2
#define WARPS_PER_BLOCK (WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N)

// Assuming maximum E value (number of matrices in M)
#define MAX_E 64  // Adjust based on your maximum E

// Declare M in constant memory (assuming E is not too large)
// __constant__ __half M_const[MAX_E][WMMA_K][WMMA_N]; // constant memory is like 64KB, so we can keep more than 100 expertparts easily


__global__ void batched_matmul_merge_fused_kernel(const __half *__restrict__ A,
                                            const __half *__restrict__ B,
                                            const __half *__restrict__ W,
                                                const __half *__restrict__ SPA,
                                            float *__restrict__ C,
                                            int M, int N, int K, int batch_size, int E)
{
    // Batch index
    int batch = blockIdx.z;
    // Thread warp index within the thread block
    int warp_id = threadIdx.x / WARP_SIZE; // warp withing block
    // int lane_id = threadIdx.x % WARP_SIZE; // thread within warp
    // blcoksize is 128, so 4 warps per block
    int warp_row = warp_id / WARPS_PER_BLOCK_N; // Integer division
    int warp_col = warp_id % WARPS_PER_BLOCK_N;

    // Compute the tile indices
    // determine the row and column indices of the tile in the output matrix C that a particular warp will compute.
    int tile_row = blockIdx.y * WARPS_PER_BLOCK_M + warp_row;
    int tile_col = blockIdx.x * WARPS_PER_BLOCK_N + warp_col;

    // Compute the starting row and column indices of the tile
    int row = tile_row * WMMA_M;
    int col = tile_col * WMMA_N;

    if (row < M && col < N)
    {   

        // Declare the accumulator fragment
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        // Load per-batch weights into registers
        __half W_reg[MAX_E];
        for (int e = 0; e < E; ++e)
        {
            W_reg[e] = W[batch * E + e];
        }

        // Loop over K dimension
        for (int k = 0; k < K; k += WMMA_K)
        {   
            // Declare and initialize fragment for A matrix
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_K, __half, wmma::row_major> a_frag;
            const __half *a_ptr = A + batch * M * K + row * K + k;

            // Declare fragment for B matrix
            wmma::fragment<wmma::matrix_b, WMMA_K, WMMA_N, WMMA_K, __half, wmma::row_major> B_frag;

            // Compute global memory address for B
            const __half *B_ptr = B + batch * K * N + k * N + col;

            // Bounds checking for K dimension
            if (k + WMMA_K <= K)
            {// Load the A matrix
                wmma::load_matrix_sync(a_frag, a_ptr, K);

                // Load the B matrix
                wmma::load_matrix_sync(B_frag, B_ptr, N);

                // Compute merged_M_frag for the current K segment
                wmma::fragment<wmma::matrix_b, WMMA_K, WMMA_N, WMMA_K, __half, wmma::row_major> merged_M_frag;
                wmma::fill_fragment(merged_M_frag, __float2half(0.0f));

                // Accumulate weighted sum into merged_M_frag
                for (int e = 0; e < E; ++e)
                {
                    // Load M[e] tile from constant memory into a fragment
                    wmma::fragment<wmma::matrix_b, WMMA_K, WMMA_N, WMMA_K, __half, wmma::row_major> M_frag;
                    const __half *M_ptr = SPA + (e * K * N) + (k * N + col);

                    // Load the M[e] tile
                    wmma::load_matrix_sync(M_frag, M_ptr, N);

                    // Multiply M_frag by W_reg[e] and accumulate
                    for (int i = 0; i < M_frag.num_elements; ++i)
                    {
                        M_frag.x[i] = __hmul(M_frag.x[i], W_reg[e]);
                    }

                    // Accumulate into merged_M_frag
                    for (int i = 0; i < merged_M_frag.num_elements; ++i)
                    {
                        merged_M_frag.x[i] = __hadd(merged_M_frag.x[i], M_frag.x[i]);
                    }
                }

                // Add merged_M_frag to B_frag
                for (int i = 0; i < B_frag.num_elements; ++i)
                {
                    B_frag.x[i] = __hadd(B_frag.x[i], merged_M_frag.x[i]);
                }

                // Perform the matrix multiplication
                wmma::mma_sync(c_frag, a_frag, B_frag, c_frag);
            }
            else
            {
                assert(false && "K must be a multiple of WMMA_K");
            }
        }

        // Store the output
        float *c_ptr = C + batch * M * N + row * N + col;
        wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
    }
}

torch::Tensor bmm_w_merge(torch::Tensor A, torch::Tensor B, torch::Tensor W, torch::Tensor SPA)
{
    const int batch_size = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);
    const int E = SPA.size(0);

    // Ensure the inputs are in half precision
    TORCH_CHECK(A.scalar_type() == at::kHalf, "A must be half-precision");
    TORCH_CHECK(B.scalar_type() == at::kHalf, "B must be half-precision");
    TORCH_CHECK(W.scalar_type() == at::kHalf, "W must be half-precision");
    TORCH_CHECK(SPA.scalar_type() == at::kHalf, "SPA must be half-precision");

    size_t M_size = E * WMMA_K * WMMA_N * sizeof(__half);
    TORCH_CHECK(M_size <= 64 * 1024, "M tensor size exceeds constant memory capacity");

    TORCH_CHECK(K % WMMA_K == 0, "K must be a multiple of WMMA_K (16)");
    // cudaMemcpyToSymbol(M_const, SPA.data_ptr<at::Half>(), M_size); // constant memory is like 64KB, so we can keep  limited number of expertperts
    
    auto C = torch::empty({batch_size, M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

    // idea:
    //  each warp computes one tile!
    //  since each block has 4 tiles, each block computes 4 tiles, 2 in each direction
    //  Calculate the number of tiles
    int M_TILES = (M + WMMA_M - 1) / WMMA_M;
    int N_TILES = (N + WMMA_N - 1) / WMMA_N;

    // Calculate grid dimensions
    int gridDimY = (M_TILES + WARPS_PER_BLOCK_M - 1) / WARPS_PER_BLOCK_M; // so this will be 8 if we have 16 tiles

    int gridDimX = (N_TILES + WARPS_PER_BLOCK_N - 1) / WARPS_PER_BLOCK_N; // this will be also 8 if we have 16 tiles

    dim3 threads(WARP_SIZE * WARPS_PER_BLOCK);
    dim3 blocks(gridDimX, gridDimY, batch_size);

    // Launch the CUDA kernel
    batched_matmul_merge_fused_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(W.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(SPA.data_ptr<at::Half>()),
        C.data_ptr<float>(),
        M, N, K, batch_size, E);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA kernel failed: " << cudaGetErrorString(err);
        throw std::runtime_error(ss.str());
    }

    return C;
}