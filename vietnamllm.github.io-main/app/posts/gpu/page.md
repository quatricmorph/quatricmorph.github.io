# How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog

> December 2022

In this post, I’ll iteratively optimize an implementation of matrix multiplication written in CUDA. My goal is not to build a cuBLAS replacement, but to deeply understand the most important performance characteristics of the GPUs that are used for modern deep learning. This includes coalescing global memory accesses, shared memory caching and occupancy optimizations, among others.

Matrix multiplication on GPUs may currently be the most important algorithm that exists, considering it makes up almost all the FLOPs during the training and inference of large deep-learning models. So how much work is it to write a performant CUDA SGEMM from scratch? I’ll start with a naive kernel and step-by-step apply optimizations until we get within 95% (on a good day) of the performance of cuBLAS (NVIDIA’s official matrix library):

| Kernel                   |  GFLOPs/s | Performance relative to cuBLAS |
| :----------------------- | --------: | :----------------------------- |
| 1: Naive                 |   `309.0` | `1.3%`                         |
| 2: GMEM Coalescing       |  `1986.5` | `8.5%`                         |
| 3: SMEM Caching          |  `2980.3` | `12.8%`                        |
| 4: 1D Blocktiling        |  `8474.7` | `36.5%`                        |
| 5: 2D Blocktiling        | `15971.7` | `68.7%`                        |
| 6: Vectorized Mem Access | `18237.3` | `78.4%`                        |
| 9: Autotuning            | `19721.0` | `84.8%`                        |
| 10: Warptiling           | `21779.3` | `93.7%`                        |
| 0: cuBLAS                | `23249.6` | `100.0%`                       |

## Kernel 1: Naive Implementation

In the CUDA programming model, computation is ordered in a three-level hierarchy. Each invocation of a CUDA kernel creates a new grid, which consists of multiple blocks. Each block consists of up to 1024 individual threads. Threads that are in the same block have access to the same shared memory region (SMEM).

The number of threads in a block can be configured using a variable normally called `blockDim`, which is a vector consisting of three ints. The entries of that vector specify the sizes of `blockDim.x`, `blockDim.y` and `blockDim.z`, as visualized below:

![CUDA_thread_hierarchy](/images/CUDA-MMM/CUDA_thread_hierarchy.png)

Similarly, the number of blocks in a grid is configurable using the `gridDim` variable. When we launch a new kernel from the host, it creates a single grid, containing the blocks and threads as specified. It’s important to keep in mind that the thread hierarchy we just talked about mostly concerns program correctness. For program performance, as we’ll see later, it’s not a good idea to treat all threads in the same block as equals.

For our first kernel, we’ll use the grid, block and thread hierarchy to assign each thread a unique entry in the result matrix C. Then that thread will compute the dot product of the corresponding row of A and column of B, and write the result to C. Due to each location of C being written to by only one thread, we have to do no synchronization. We’ll launch the kernel like so:

```cpp
// create as many blocks as necessary to map all of C
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
// 32 * 32 = 1024 thread per block
dim3 blockDim(32, 32, 1);
// launch the asynchronous execution of the kernel on the device
// The function call returns immediately on the host
sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

CUDA code is written from a single-thread perspective. In the code of the kernel, we access the `blockIdx` and `threadIdx` built-in variables. These will return different values based on the thread that’s accessing them.

<!-- ```cuda -->

```cpp
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
```

To visualize this simple kernel:

![naive](/images/CUDA-MMM/naive-kernel.png)

This kernel takes about 0.5s to process three 4092² fp32 matrices on my A6000 GPU. Let’s do some non-implementation-specific calculations:

### Lower Bounding the Fastest Possible Runtime

For a matrix multiplication of two 4092² matrices, followed by an addition of a 4092² matrix (to make the [GEMM](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3)):

1. Total FLOPS: `2*4092³ + 4092² = 137 GFLOPS`
2. Total data to read (minimum!): `3 * 4092² * 4B = 201MB`
3. Total data to store: `4092² * 4B = 67MB`

So 268MB is the absolute minimum of memory that any implementation would have to transfer from/to global GPU memory, assuming it has a big enough cache. Let’s calculate some upper bounds on kernel performance. The GPU is advertised with 30TFLOPs/s of fp32 compute throughput and 768GB/s of global memory bandwidth. If we achieved those numbers, we’d need 4.5ms for the calculation and 0.34ms for the memory transfers. So in our napkin math, the calculation takes ~10x more time than the memory accesses. This means our final optimized kernel will be compute-bound, as long as we end up having to transfer <10x the absolute minimum memory volume of 278MB.

Now that we’ve calculated some lower bounds for our fp32 GEMM calculation, let’s get back to the kernel on hand, to figure out why it’s so much slower than it could be.

### Memory Access Pattern of the Naive Kernel

In our kernel, two threads in the same block with ThreadIds (0, 0) and (0, 1) will load the same column of B but different rows of A. If we assume the worst case of zero caching, then each thread has to load `2*4092+1` floats from global memory. As we have 4092² threads total, this would result in 548GB of memory traffic.

Below is a visualization of the memory access pattern of our naive kernel, taking two threads A (red) and B (green) as an example:

![naive_kernel_mem_access](/images/CUDA-MMM/naive_kernel_mem_access.png)

So to recap, when I run this kernel on an A6000 GPU it achieves ~300GFLOPs when multiplying two 4092x4092 float32 matrices. Pretty bad, considering that the A6000 is advertised as being able to achieve almost 30 TFLOPs. So how can we start to make this faster? One way is to optimize the memory access pattern of our kernel such that global memory accesses can be coalesced (=combined) into fewer accesses.

## Kernel 2: Global Memory Coalescing

Before we get into global memory coalescing, we need to learn about the concept of a warp. For execution, the threads of a block are grouped into so-called warps, consisting of 32 threads. A warp is then assigned to a warp scheduler, which is the physical core that executes the instructions. There are four warp schedulers per multiprocessor. The grouping into warps happens based on a consecutive `threadId`. If we set the `blockDim` to be multi-dimension, then the threadId is calculated like so:

```cpp
threadId = threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z)
```

Then, threads with neighbouring `threadId` become part of the same warp. Below I tried to illustrate this, using a smaller “warpsize” of 8 threads (real warps always contain 32 threads):

![threadId_to_warp_mapping](/images/CUDA-MMM/threadId_to_warp_mapping.png)

The concept of a warp is relevant for this second kernel, as sequential memory accesses by threads that are part of the same warp can be grouped and executed as one. This is referred to as **global memory coalescing**. It’s the most important thing to keep in mind when optimizing a kernel’s GMEM memory accesses toward achieving the peak bandwidth.

Below is an example, where consecutive memory accesses by threads in the same warp are grouped, allowing each warp to execute 8 memory accesses using only 2 32B loads:

![GMEM_coalescing](/images/CUDA-MMM/GMEM_coalescing.png)

In reality, the GPU supports 32B, 64B and 128B memory accesses. So, if each thread is loading a 32bit float from global memory, the warp scheduler (probably the MIO) can coalesce this `32*4B=128B` load into a single transaction. This is only possible if the floats loaded are consecutive in memory, and if access is aligned. If they aren’t, or if access cannot be coalesced for some other reason, then the GPU will execute as many 32B loads as necessary to fetch all floats, leading to a lot of wasted bandwidth. Profiling our naive kernel, we can observe the detrimental effect of non-coalesced access as we achieve only 15GB/s of GMEM throughput.

Looking back at the previous kernel, we assigned threads their entry of C like so:

<!-- ```cuda -->

```cpp
const uint x = blockIdx.x * blockDim.x + threadIdx.x;
const uint y = blockIdx.y * blockDim.y + threadIdx.y;
```

Hence, threads of the same warp (those with consecutive `threadIdx.x`) were loading the rows of A non-consecutively from memory. The naive kernel’s pattern of accessing the memory of A looked more like so:

![Naive_kernel_mem_coalescing](/images/CUDA-MMM/Naive_kernel_mem_coalescing.png)

To enable coalescing, we can change how we assign positions of the result matrix C to threads. This change in the global memory access pattern is illustrated below:

![Naive_kernel_improved_access](/images/CUDA-MMM/Naive_kernel_improved_access.png)

To implement this, we only need to change the first two lines:

<!-- ```cuda -->

```cpp
const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

if (x < M && y < N) {
  float tmp = 0.0;
  for (int i = 0; i < K; ++i) {
    tmp += A[x * K + i] * B[i * N + y];
  }
  C[x * N + y] = alpha * tmp + beta * C[x * N + y];
}
```

And we call it like so:

```cpp
// gridDim stays the same
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
// make blockDim 1-dimensional, but don't change number of threads
dim3 blockDim(32 * 32);
sgemm_coalescing<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

Global memory coalescing increases memory throughput from 15GB/s to 110GB/s. Performance reaches 2000 GFLOPS, a big improvement compared to the 300 GFLOPS of the first, naive kernel. For the next kernel, we’ll use the GPU’s fast on-chip memory, called shared memory, to cache data that will be re-used.

## Kernel 3: Shared Memory Cache-Blocking

Next to the large global memory, a GPU has a much smaller region of memory that is physically located on the chip, called shared memory (SMEM). Physically, there’s one shared memory per SM. Logically, this shared memory is partitioned among the blocks. This means that a thread can communicate with the other threads in its block via the shared memory chunk. On my A6000 GPU, each block has access to a maximum of 48KB of shared memory.

As the shared memory is located on-chip, it has a much lower latency and higher bandwidth than global memory. I couldn’t find good benchmark results for the Ampere architecture but for Volta (released in 2017) the benchmarks performed in [this paper](https://arxiv.org/abs/1804.06826) report 750GiB/s of global memory bandwidth, and 12,080GiB/s of shared memory bandwidth.

So for this next kernel, we’ll load a chunk of A and a chunk of B from global memory into shared memory. Then we’ll perform as much work as possible on the two chunks, with each thread still being assigned one entry of C. We’ll move the chunks along the columns of A and the rows of B performing partial sums on C until the result is computed.

This is illustrated below:

![cache](/images/CUDA-MMM/cache-blocking.png)

The important parts of the code are below, with variable names corresponding to the plot above:

<!-- ```cuda -->

```cpp
// advance pointers to the starting positions
A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
B += cCol * BLOCKSIZE;                        // row=0, col=cCol
C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

float tmp = 0.0;
// the outer loop advances A along the columns and B along
// the rows until we have fully calculated the result in C.
for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
  // Have each thread load one of the elements in A & B from
  // global memory into shared memory.
  // Make the threadCol (=threadIdx.x) the consecutive index
  // to allow global memory access coalescing
  As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
  Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

  // block threads in this block until cache is fully populated
  __syncthreads();

  // advance pointers onto next chunk
  A += BLOCKSIZE;
  B += BLOCKSIZE * N;

  // execute the dotproduct on the currently cached block
  for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
    tmp += As[threadRow * BLOCKSIZE + dotIdx] *
            Bs[dotIdx * BLOCKSIZE + threadCol];
  }
  // need to sync again at the end, to avoid faster threads
  // fetching the next block into the cache before slower threads are done
  __syncthreads();
}
C[threadRow * N + threadCol] =
    alpha * tmp + beta * C[threadRow * N + threadCol];
```

This kernel achieves ~2200 GFLOPS, a 50% improvement over the previous version. We’re still far away from hitting the ~30 TFLOPs that the GPU can provide. This is obvious from the roofline plot below:

![roofline_kernel_3Roofline analysis of kernel 3](/images/CUDA-MMM/roofline_kernel_3.png)

At a CHUNKSIZE of 32, this uses `2*32*32*4B=8KB` of shared memory space. My A6000 GPU has a maximum of 48KB of shared memory space available for each block, so we’re far away from hitting that limit. This is not necessarily a problem, as there are downsides to increasing per-block shared-memory usage. Each multiprocessor (SM) has a maximum of 100KB of SMEM available. This means that if we’d modify our kernel to use the full 48KB of SMEM available, each SM could only keep two blocks loaded at the same time. In CUDA parlance, increasing per-block SMEM utilization can decrease [occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy). Occupancy is defined as the ratio between the number of active warps per SM and the maximum possible number of active warps per SM.

High occupancy is useful because it allows us to hide the high latency of our operations, by having a bigger pool of issue-able instructions available. There are three main limits to keeping more active blocks loaded on an SM: register count, warp count and SMEM capacity. Let’s do an example calculation for our current kernel.

### Occupancy Calculation for Kernel 3

Here are the relevant hardware stats for my GPU, obtained from the `cudaGetDeviceProperties` API (Multiprocessors are the SMs we talked about earlier):

| Metric                                     | Value            |
| :----------------------------------------- | :--------------- |
| Name                                       | NVIDIA RTX A6000 |
| Compute Capability                         | 8.6              |
| max threads per block                      | 1024             |
| max threads per multiprocessor             | 1536             |
| threads per warp                           | 32               |
| warp allocation granularity                | 4                |
| max regs per block                         | 65536            |
| max regs per multiprocessor                | 65536            |
| reg allocation unit size                   | 256              |
| reg allocation granularity                 | warp             |
| total global mem                           | 48685 MB         |
| max shared mem per block                   | 48 KB            |
| CUDA runtime shared mem overhead per block | 1024 B           |
| shared mem per multiprocessor              | 102400 B         |
| multiprocessor count                       | 84               |
| max warps per multiprocessor               | 48               |

And here are the resource demands for our kernel:

|                      |        |
| :------------------- | :----- |
| Registers per Thread | 37     |
| SMEM per Block       | 8192 B |
| Threads per Block    | 1024   |

Work is scheduled onto the SMs on a block granularity. Each SM will load more blocks, as long as it has enough resources to accommodate them. Calculation:

- **Shared memory**: 8192B/Block + 1024B/Block for CUDA runtime usage = 9216B/Block. (102400B per SM) / (9216B per Block) = 11.11 ⇒ 11 Blocks upper limit.
- **Threads**: 1024 Threads per Block, max 1536 threads per SM ⇒ Upper limit 1 block.
- **Registers**: 37 regs per thread _ 32 threads per warp = 1184 regs per warp. Register allocation granularity is 256 regs on a warp level, hence rounding up to 1280 regs per warp. We have (1024 threads / 32) = 32 warps per block, hence 1280 regs per warp _ 32 warps per block = 40960 regs per block. Max 65536 regs per SM ⇒ upper limit 1 block.

So this kernel is limited by the number of threads per block, and the number of registers per thread. We cannot load more than one block per SM, giving us a final occupancy of 32 active warps / 48 max active warps = 66%.

A 66% occupancy is not too bad, so this doesn’t explain why our kernel runs so slow. Looking at the profiler gives us some hints. First, if we look at the mix of executed instructions, most of them are memory loads:

![kernel_3_profiler_instr_mix](/images/CUDA-MMM/kernel_3_profiler_instr_mix.png)

Our inner loop looks like this in PTX ([Godbolt link](https://godbolt.org/#z:OYLghAFBqd5TKALEBjA9gEwKYFFMCWALugE4A0BIEAZgQDbYB2AhgLbYgDkAjF%2BTXRMiAZVQtGIHgBYBQogFUAztgAKAD24AGfgCsp5eiyahUAV0wtyKxqiIEh1ZpgDC6embZMQAVgBs5M4AMgRM2AByngBG2KQgAMwA7OQADuhKxA5Mbh5evgFpGfZCIWGRbDFxSdbYtsVMIkQspEQ5nt7SyTbYdlmNzUSlEdGxCV1NLW15ndYTg6HDFaNJAJTW6GakqJxcAKQATPGhqB44ANS78S4SwGTESGyXuLtaAIIHR0wnFtgXV6hKIiEdBPF7vQ7HU6/S4uAFA%2BgEKKgt4fSE/P6wsxRIxKAD6ADd9gA6JDI8Gfb7nGHmSy40hmYQEDgkslgj44OhhM4uXAASSCuIAIryAGoQACy5DO4RWZyg4tlBwAQnKZQBaHiygD0qpWbLeuNxwHo6CiEkNZ3x6AImDOSmA2DYbFxSiQzWwmFxHGd2PQqAA1hBQkQzpKzsHpVKIwBpKU0E0sEMSFJu8hgs4ZzNZ7M53N5zMYJiAs7x9CJs4AKleUsLxdL5YrSrTb3zrbb%2BfrIZiTTjCZDFZcisSSvTZ1rIYjSqCAHkXNGRLyAFrPeKC%2BL7S4jlsXbdanVEJC/DZEFJmLsmgNnA/lgDuvxvxhDJDH6DYp6Iv1CV6QBCU39I2AsJgvoBqO45nGYEaoAASugN5/IKZwgf6vKYOoRLqJuYFCMWkHCGOOQIUhF4oWhRIAJ5YSiu46hIF6Jr8URmDQNCxCWZBjpsAH4ch4ZMCWLDFq67q2t6o57naboAaJjrhn%2BwnSUh2BEHezBnHR/6AZgf5fiwxF%2Bv6o6GgpHoWp2ZyvEouw%2BEqU6zvOS64JWZx2XOC7LtZgpUa8GbGVJpm4iWfYuVZNmuQ5y7OeF7nPD4XnxFu7w0d%2Bn5MGEpBnKQcEXPsfgvvQ363tgYBcAB6moNsSgZCYfHfr%2BmlAdhRYhnhT5IABQGEZciEHh1mCoeh6g5T4Lkzm5jneRm4GtQ1mCwfB3WzQNGFnDq0UTQl%2Bo%2BattGYPixjbGcaTBrEf7PgevyAgMoTAEd6SZDho4%2Bcqi0wdlFajfZMXOdGm7tnmElZTe3VvTeNbuN1WijiqL2rgR7hRWNEUriq/2toDcGQ%2BD9Agzko4uDlI5w6DiNfY5znhIT8MFR960eQlO2ZZjq6g9juMQ9R23mUQb5EVoRJQ5t26CBlQY8aRQ2LYLKpRBLGJnL9DOywNhOLXTuBDolWYSQAEiw%2BK/IBqBILNZwJraQhHjQKVnLUjrMEQOn8c9uUuaOGYSeKLD%2Br8F2zYREDdb1WnLeosp%2B7WPRngQBt8TgmHbh7%2B7oOp9AmvBxqmhIZzemQ5HlZVf4YBI2BKKgN3uxZoVKsHQHzaT42Rcq/sQ3FRHvDZtdze9CtU13eNxZNmZKtXXf17TSPfc3/et4havWTX7VaeP0p90vnWz95lcSbxXdO3Vf68Qy9gFeIxufn%2BNBmGn%2BdpKeRgfpglfGeRXx7xAepC9tGYu0TiHq0PDMMN9h/0%2Bo3JyH1whb0TozbA6go4fhtpgdAJ4sqYDMHYM4QgbbmFINxIg9B85n0PLaZClcRZygjMgogKspZ/WobQq4YDkZ/ReiAhhaFNaVwzDzFIqs4aWQXmPHu6sqYcMwm3Cs3C8wjwXuIhuyM159QHvFLWmZdiJC8jAiSYQPRXhTkoV%2BqB1LABYLpNqhsmCYClM%2BfW1pbQ0EEh%2BDKe9t46lYkQY2N0bZhHUOeAyfFzqHjHCwc%2BSkRaXXTmxPe6kyrILCM/F0Rj36fzURorR20XBCPXt3eCkClFaRUQhaRyY3TOV4VTbsekPpZM7jklelNp45JUVvTRXA1j0G4D4fg3guA6HIOgbgLgFCCh8lkkBzclAbC2NCQ4fByBEG0O0tYh4gKjA/uQf0vgtCGG4NIHpSyBncH4EoEAOzFl9PaeQOAsAUAYDfAwWIlBqD3JSI8uITB8QVR4DwZIOB8QEG2CKAg2AbzThSMwQ5dB6DONORAKIhyoihGaORbg8z7kcGENOJghDDk4DYMYY0Ox%2BmEAAr0A2pzLmBHgeYD8aL%2BAnU6VShEURSAorcDgQ5RBSBMnpWseMLBgBKGBaC8FkLeD8EEMIMQJcpCyClYoFQGhDn6H2IYQlaALBWBZacyAax0ApHqJSk5dteiOAgM4KY3geCBCsUMcolQDCFAetkdw7QnX3XqPakYcQbXdDNQ0OYVqDD%2BvqP0Fo3qli%2BtmAMYNfq5iRsdZqdYmxthSA6V0g5VLBlcGlCKFwBMflEkSALOU%2BBiAcQ%2BJqfgFydArBWVpdZawtk%2BB2Uy/Z5Ben9JzScs5Cyln1t2VwfY/A2AgGkPzPwPhEgAE5DhaH2IkeIWgfAzp8D4Tthye39suWsG5yAQD/MBdgZ5EBXnvPCOwHY4R82Fp4MWgW/APQVp5ZgAwCqZWSBkHIYQyg1CaCpaqmodQshOCscG6QAAOW1mBE2jB4D4G1zr6hxtSJ6rIcHfXSBncBnoYag1uumNB0NfQE0LAdfBxDMbJiEetdR%2BYZQfVypnWsbl2BsA2jORmrg3TN3Zu4IKbAALDoirvBlG9BazhFpLVoMthASAZSrVKNwDzGCKcOPsFYNaB0NrWXEDZWyeBaDbXs0dvgZ1EniNIfYWhEh%2BBnYkXKM6J2yC7fwbdpzzk6c2VIYzQ74hZu7ccndda1gG1IBkRw0ggA%3D)):

```c
ld.shared.f32   %f91, [%r8+3456];
ld.shared.f32   %f92, [%r7+108];
fma.rn.f32      %f93, %f92, %f91, %f90;
```

That’s not good, given that a memory load is bound to have a higher latency than a simple FMA, and given that we know our kernel should be compute bound. We see this effect when looking at the profiler’s sampling of warp states. This quantifies how many cycles were spent in each state per executed instruction:

![kernel_3_profiler_warp_stalls](/images/CUDA-MMM/kernel_3_profiler_warp_stalls.png)

The meaning of the states is documented in the [Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference). For `Stall MIO Throttle` it reads:

> Warp was stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions

We’re not using special math instructions, nor dynamic branches, so it’s clear that we’re stalling waiting for our SMEM accesses to return. So how do we make our kernel issue less SMEM instructions? One way is to have each thread compute more than one output element, which allows us to perform more of the work in registers and relying less on SMEM.

## Kernel 4: 1D Blocktiling for Calculating Multiple Results per Thread

So this next kernel works like our last kernel, but adds a new inner loop, for calculating multiple C entries per thread. We now use a SMEM cache size of `BM*BK + BN*BK = 64*8 + 64*8 = 1024` floats, for a total of 4KB per block. Below a visualization. I have highlighted two of the threads and the values they access in the inner loop in orange and red.

![kernel_4_1D_blocktiling](/images/CUDA-MMM/kernel_4_1D_blocktiling.png)

All of the important changes for this kernel happen in the inner loop. The loading for GMEM to SMEM stays largely the same as before. Let’s have a look:

```c++
// allocate thread-local cache for results in registerfile
float threadResults[TM] = {0.0};

// outer loop over block tiles
for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
  // populate the SMEM caches (same as before)
  As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
  Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
  __syncthreads();

  // advance blocktile for outer loop
  A += BK;
  B += BK * N;

  // calculate per-thread results
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // we make the dotproduct loop the outside loop, which facilitates
    // reuse of the Bs entry, which we can cache in a tmp var.
    float Btmp = Bs[dotIdx * BN + threadCol];
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
      threadResults[resIdx] +=
          As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
    }
  }
  __syncthreads();
}
```

This kernel achieves ~8600 GFLOPs, 2.2x faster than our previous kernel. Let’s calculate how many memory accesses each thread performed in our previous kernel, where each thread calculated one result:

- GMEM: K/32 iterations of outer loop \* 2 loads
- SMEM: K/32 iterations of outer loop _ BLOCKSIZE (=32) _ 2 loads
- Memory accesses per result: K/16 GMEM, K\*2 SMEM

And for our new kernel, where each thread calculates eight results:

- GMEM: K/8 iterations of outer loop \* 2 loads
- SMEM: K/8 iterations of outer loop _ BK(=8) _ (1 + TM(=8))
- Memory accesses per result: K/32 GMEM, K\*9/8 SMEM

As expected, we now spend much fewer cycles per instruction stalling due to memory pressure:

![Kernel_4_profiler_warp_stalls](/images/CUDA-MMM/Kernel_4_profiler_warp_stalls.png)

### Sidenote on Compiler Optimizations

Above we explicitly cached the entry of B into `Btmp` and reordered the two inner loops for efficiency. If we don’t do that, then the code looks like this:

```c++
for (uint resIdx = 0; resIdx < TM; ++resIdx) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    threadResults[resIdx] +=
      As[(threadRow * TM + resIdx) * BK + dotIdx] *
      Bs[dotIdx * BN + threadCol];
  }
}
```

Interestingly, this has no adverse effect on performance. This is surprising since our inner two loops now incur BK (=8) _ TM (=8) _ 2 = 128 SMEM accesses, instead of the previous 72. Looking at the assembly ([Godbolt link](https://godbolt.org/#z:OYLghAFBqd5TKALEBjA9gEwKYFFMCWALugE4A0BIEAZgQDbYB2AhgLbYgDkAjF%2BTXRMiAZVQtGIHgBYBQogFUAztgAKAD24AGfgCsp5eiyahUAV0wtyKxqiIEh1ZpgDC6embZMQAVnLOAGQImbAA5TwAjbFIQAHYtcgAHdCViByY3Dy9fJJS0oSCQ8LYomPjrbFt7IREiFlIiTM9vPxtsO3Ta%2BqJCsMjouISlOoam7NaRnuC%2BkoH4gEprdDNSVE4uAFIAJgBmYNQPHABqDZ2XCWAyYiQ2U9wNrQBBbb2mA4tsE7PxJRUGu4ez12%2B0On1OLlQw0I6ABTxeII%2BXwhUPoBAisKBr3ex3B5giRiUAH0AG5bAB0SAx8LeoKR5kshNIZmEBA4FKpcN2ODoISOLlwAEkAoSACICgBqEAAsuQjqF5kcoFKFdsAEKK%2BUAWh4CoA9Br5oDAYTCcB6OgIhITUdiegCJgjkpgNg2GweCKAOr1RIAFQYwWAEGCRCOMqOwblsojAGlZTRzSwQxJEkgrICjhnM1nsznczmMExhkd4%2BhE0cAFSPWUFoslsvl1VxhMhqJ1cjpvOdrs5ush8suFWxVVGp4ZmshiOqqVfEVHABs0lOw9HR3H4eER1VoRn88XO2XjzHQiLk%2BjO4AHEuO2uIz7p6dZ5f9x3dfqBTQjgB3T7xgiJI7qEcxgOgAnkcSDRJ835HM6IYAH47Fo2w%2BEcjC/EciTRIIpBsMYazFmQqH1M6Ry4UQpAEGsShki%2B%2Bo%2BhBq4rKQzBELKiHIcWLDDNEq5CHQwArIm6RHMwSgrNgShHEQqYtuaqAANaSZ%2B1yOtgACOZgsQQEi0Uc%2BLoApAoipJLCoFRkkYFkhZHOgH7qioGlaRI9AgbKn5IAwnxKKmFEmFJDFKOwnykOgn42R%2Bjw0Sur5HPRXnmt%2BpC8Uw/GCdUTBfss9AOt59SfJZzSSbZRxVnpZhJmZEmScG6CbplZjZbpURHEwQiag5mkshIZJHCItXST%2BXFEDxBapaQQlCOBXF6dgRDDUlSiJEJEioQZEjECBunAeBzD4SwMHMYmPEBFs4HEEc43DVFh7JUWZgRqgABKoU7vphmYOoZKbc%2BK5rvdG6oJkr1yfJAofWS6hXnCv3HiG/0hiQdT0I9EkNUQSiqiD9iMDuU4VpuoRQzdMWHtJh0OgQknMYtx5ojj2GrhI5hGPYfl3iJjAcMI1UZQNelY5516w0c8MtZ49HkxjAs4w%2BUnoEjKNifQ6OYwZ8nY58dFSkTumK2jSiqNEqsKUc%2Bp68rBvRBL2AsA6D6y9btuW6QxvyR2XF/EQEBMOLSCS67GszrLb3ySKrIQ4aP3PNF%2B%2Bo26gSBft6X4MPQjP0MzR1HDsWzluzlQuixShuSp2czQG/n5e4zT1EchBsNdR6FhOG5kzbrjuDure22Dn2ARxW5E43J4t37bfPWFstd5gPcQ6bBM6zHQH0HJmeLaZP6EXz5ikMxG4h4HwS9VKuDayuJq5cxmDWr2JVKBsPiqnj5abtG98ioPRznz52BX4SxbNpuO%2BD9VRnmfgPHw78o66SlOgYknx96eTljNYAwQmDl2Ko8MAXAqYvW2qqLBFkq5eA7IeNUssnovTAdOZ%2Br8o4ZnVGQnYs5AYdzAYTOhfIThbGHEw1c498ZPzlFw9ULDU5sIXsTfUJBMDoBAEBTAugzBFmkpTJBccE5TyQQmB0bA0Z/hxixCiEkgJMEwE1Wa80RLqESOac6W9TIBQIAALwku7X40QvaCJAUHXhIcw713UJHA8GYPYeIgFuARZ57a%2BJBv4iOH8/oRlQdETIpDeFTxnn3LYKEQFLlNvqT83pNSMDganAA4sfacGAJASVQAGIWTcRZJKYCEUg480mzgyeDQC%2BpckcMSRuZJpBMgMPSaPbu3SuE5PYeqGKhTSCJGKdgUpRwKkn14jUpQdSTANLus01p49RmdPGdPSZvSZkjkkUvFew1/Lk2KWtVO4h44byStTfW64LrYBQdxUgdBGAdhvlPc26N76qjvG/HcGwhxaDJEhWIkCDy6WWJY806B/ywJ4iHKSnk74rgZhAUWERQYfR3EhfcekSV9zOEcWh6piU92EbLEBg5gmZhiskRIDVM58xEJUxmLy8U3QzI8IBqohntMicI9crTUmQtls8B%2BErQqHhodKoZcqIEf3oWK5Vn51RsPVS0lJ7hhwQNxmCvVBqhFqhlSa%2BgZrEUdgzOfECbwp5KAgEEy5WYYq22JHheB0tsDOpKky3hfS2X0PDbObxz8LnQ2FQRJKhKIzU0ZbLcl6p02kvBLFbWFKyHcJzYEk4Q5Q0ZgJaLGRRAM28KzbXeWdaXAvzyUW1UNae6sorZmYFqMLZgpLfK7h9sVxdlFWCiAwLKH5ulSWhUYConcMbbWj6kKwFis7aSw1tqp6ZDftqzM0L35jrLSepNrr3UnM9d6lcx6fX5K/BRW5KKK5fKVqC/FhFU0bhLWSvJf680QsLSO4tEku1lqjXySd06wrP3Zra%2Bd%2BNty7pOfu81o6k1ZmTKmfGfaP1iqHRhkdp6sytn2s/FwMGTn8Pg/eZdSH43Sr3e4A9HD71wgRVwRY9BuA%2BH4N4LgOhyCyK4C4BQIpDxUdA/ZZYqwwS7D4OQIg2huOLAgrbAYXryDyV8AkXjXBpD8DYCAaQ0gyTnkszsOcNmtixGkAATniAucggnhOif4EoEACQVNCe4%2BQOAsAUAYDYIkTyFAqAQBC2FxgMQmDEjMjwHgsRyA4GJJRbA4oCDYE/AAeUwm5/g/z5peYgBEVT5AIjBHqCBbgSmQtcyILlpgLkKs4FwiYSQfnUsEGYh0OBXnuvYHUO0cq6wlPBkqBV1EERxqkBAm4HAFXyKsjq/5%2BMLBgBKCyzl/LzA1tyGEGIGpUhZCCGEMoNQmhuv6C2IYYwpgLBWBm15yAix0XpUG5qXLOwjiahFKEEUuBVQKDKX9zUzpWlHU1BgHAD56jxwfNFsbhJzxznILDkNTCwXI%2BGqj9HSg2D48hZqTUiQiDqC4pqD76Q75MM1MSP7UIkftuS55/OHRHAQGcGMbwPB/CmN6MUUoBhkipHSrz0XeR0pC/6DEfnbROdMC6KMIhfOKhVE6JMWXsx5fWEmJLhX2vpjC4GDqJYKw1gGEKcIfLtaUqib4DxvjAmKuiblOKFwLaktkliHCxU%2BBiCEReDqfgvmdDzHU23LTzvDPGZADsHgZIeBbDs%2BeHw54zOxEz7EHYrm3fcE8955TqnFiBeQCAFFXLWKRei%2BF0IQVuChE997pPfvhM/yDxRTABhzuiHEJIGQh3FAqA0BV275BPzjUSGt2P/H8/ddE7l8q1fwoe690cH37fFRuFC%2BFrh2d5hh9L7Hoz5ATO7DJA5yzOefB2diDwc8WwtApcKyJwv1hi/h7UzpvThhuA7Cu6L4f7f5l6IAQAoBpYZaUDUB16xYN4cBN4t6b5t5wr8Cd4kDd697yDHaD5nbyCXZj43YgB3aK7pROCmKG4C6YA64i785i75AZBq5S7i7pC0Fm4a7tDpQq6NDMEK4c7cHG5FBy4GDDDdBUFiENDsHy6LDkTYDYD2jeZz5AHubcAijLIZZHA7aJTr6t6%2B7%2B4QCB6YEH7858joB76xYH5bBH4l5%2BaR7kAaY4AxDaa6Y%2BD6bcBn5v4eaf4%2BYn7/5cBbDx48BaAJBeEgF%2BFwKkCpCODSBAA%3D)) has the answer:

```c
// first inner-most loop
ld.shared.f32   %f45, [%r9];
ld.shared.f32   %f46, [%r8];
fma.rn.f32      %f47, %f46, %f45, %f212;
ld.shared.f32   %f48, [%r9+256];
ld.shared.f32   %f49, [%r8+4];
fma.rn.f32      %f50, %f49, %f48, %f47;
ld.shared.f32   %f51, [%r9+512];
ld.shared.f32   %f52, [%r8+8];
fma.rn.f32      %f53, %f52, %f51, %f50;
ld.shared.f32   %f54, [%r9+768];
ld.shared.f32   %f55, [%r8+12];
fma.rn.f32      %f56, %f55, %f54, %f53;
ld.shared.f32   %f57, [%r9+1024];
ld.shared.f32   %f58, [%r8+16];
fma.rn.f32      %f59, %f58, %f57, %f56;
ld.shared.f32   %f60, [%r9+1280];
ld.shared.f32   %f61, [%r8+20];
fma.rn.f32      %f62, %f61, %f60, %f59;
ld.shared.f32   %f63, [%r9+1536];
ld.shared.f32   %f64, [%r8+24];
fma.rn.f32      %f65, %f64, %f63, %f62;
ld.shared.f32   %f66, [%r9+1792];
ld.shared.f32   %f67, [%r8+28];
fma.rn.f32      %f212, %f67, %f66, %f65;
// second inner-most loop
ld.shared.f32   %f68, [%r8+32];
fma.rn.f32      %f69, %f68, %f45, %f211;
ld.shared.f32   %f70, [%r8+36];
fma.rn.f32      %f71, %f70, %f48, %f69;
ld.shared.f32   %f72, [%r8+40];
fma.rn.f32      %f73, %f72, %f51, %f71;
ld.shared.f32   %f74, [%r8+44];
fma.rn.f32      %f75, %f74, %f54, %f73;
ld.shared.f32   %f76, [%r8+48];
fma.rn.f32      %f77, %f76, %f57, %f75;
ld.shared.f32   %f78, [%r8+52];
fma.rn.f32      %f79, %f78, %f60, %f77;
ld.shared.f32   %f80, [%r8+56];
fma.rn.f32      %f81, %f80, %f63, %f79;
ld.shared.f32   %f82, [%r8+60];
fma.rn.f32      %f211, %f82, %f66, %f81;
// ... continues like this for inner-loops 3-8 ...
```

The compiler unrolls both loops and then eliminates the repeated SMEM loads of the `Bs` entries, so we end up with the same amount of SMEM accesses as our optimized CUDA code.

When the PTX is compiled to SASS, the SMEM loads from `Bs` are vectorized:

```c
LDS     R26, [R35.X4+0x800] // a 32b load from As
LDS.128 R8,  [R2]           // a 128b load from Bs
LDS.128 R12, [R2+0x20]
LDS     R24, [R35.X4+0x900]
LDS.128 R20, [R2+0x60]
LDS     R36, [R35.X4+0xb00]
LDS.128 R16, [R2+0x40]
LDS.128 R4,  [R2+0x80]
LDS     R38, [R35.X4+0xd00]
```

### Areas of Improvement: Arithmetic Intensity

Our current kernel still suffers from the same stalling-for-memory problem as kernel 3, just to a lesser extent. So we’ll just apply the same optimization again: computing even more results per thread. The main reason this makes our kernel run faster is that it increases arithmetic intensity. Below I tried to make it more immediately obvious why calculating more results per thread raises arithmetic intensity:

![raising_arith_inten](/images/CUDA-MMM/raising_arith_inten.png)

In conclusion, all our kernels perform the same number of FLOPs, but we can reduce the number of GMEM accesses by calculating more results per thread. We’ll continue optimizing arithmetic intensity for as long as we’re still memory bound.

## Kernel 5: Increasing Arithmetic Intensity via 2D Blocktiling

The basic idea for kernel 5 will be to compute a grid of 8\*8 elements of C per thread. The first stage of the kernel is for all threads to work together to populate the SMEM cache. We’ll have each thread load multiple elements. This code looks like so:

```c++
for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
  As[(innerRowA + loadOffset) * BK + innerColA] =
      A[(innerRowA + loadOffset) * K + innerColA];
}
for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
  Bs[(innerRowB + loadOffset) * BN + innerColB] =
      B[(innerRowB + loadOffset) * N + innerColB];
}
__syncthreads();
```

Now that the SMEM cache is populated, we have each thread multiply its relevant SMEM entries and accumulate the result into local registers. Below I illustrated the (unchanged) outer loop along the input matrices, and the three inner loops for the dot product and the `TN` and `TM` dimension:

![kernel_5_2D_blocktiling](/images/CUDA-MMM/kernel_5_2D_blocktiling.png)

The interesting parts of the code look like this:

```c++
// allocate thread-local cache for results in registerfile
float threadResults[TM * TN] = {0.0};
// register caches for As and Bs
float regM[TM] = {0.0};
float regN[TN] = {0.0};

// outer-most loop over block tiles
for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
  // populate the SMEM caches
  for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
    As[(innerRowA + loadOffset) * BK + innerColA] =
        A[(innerRowA + loadOffset) * K + innerColA];
  }
  for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
    Bs[(innerRowB + loadOffset) * BN + innerColB] =
        B[(innerRowB + loadOffset) * N + innerColB];
  }
  __syncthreads();

  // advance blocktile
  A += BK;     // move BK columns to right
  B += BK * N; // move BK rows down

  // calculate per-thread results
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // load relevant As & Bs entries into registers
    for (uint i = 0; i < TM; ++i) {
      regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
    }
    for (uint i = 0; i < TN; ++i) {
      regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
    }
    // perform outer product on register cache, accumulate
    // into threadResults
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
      for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
        threadResults[resIdxM * TN + resIdxN] +=
            regM[resIdxM] * regN[resIdxN];
      }
    }
  }
  __syncthreads();
}
```

In the inner loop, we can reduce the number of SMEM accesses by making `dotIdx` the outer loop, and explicitly loading the values we need for the two inner loops into registers. Below is a drawing of the `dotIdx` loop across time, to visualize which SMEM entries get loaded into thread-local registers at each step:

![kernel_5_reg_blocking](/images/CUDA-MMM/kernel_5_reg_blocking.png)

Resulting performance: 16TFLOPs, another 2x improvement. Let’s repeat the memory access calculation. We’re now calculating `TM*TN = 8*8 = 64` results per thread.

- $GMEM: $K/8$ (outer loop iters) _ 2 (A+B) _ 1024/256 (sizeSMEM/numThreads) loads$
- $SMEM: $K/8$ (outer loop iters) _ 8 (dotIdx) _ 2 (A+B) \* 8 loads$
- Memory accesses per result: $K/64 GMEM$, $K/4 SMEM$

Slowly performance is reaching acceptable levels, however, warp stalls due to memory pipeline congestion are still too frequent. For kernel 6 we’ll take two measures to try to improve that: Transposing `As` to enable auto-vectorization of SMEM loads, and promising the compiler alignment on the GMEM accesses.

## Kernel 6: Vectorize SMEM and GMEM Accesses

The first optimization that I already hinted at earlier is to transpose `As`. This will allow us to load from `As` using vectorized SMEM loads (`LDS.128` in SASS). Below the same visualization of the three inner loops as for kernel 5, but now with `As` transposed in memory:

![kernel_6_As_transpose](/images/CUDA-MMM/kernel_6_As_transpose.png)

Looking at the assembly we see that loading `As` into the registers, which used to be a 32b `LDS` load, is now also a 128b `LDS.128` load, just like it had already been for `Bs`. This gives us a 500GFLOPs speedup, or ~3%.

Next, we’ll vectorize all loads and stores from/to GMEM using [vector datatypes](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/), namely `float4`.

The code looks like this:

```c++
float4 tmp =
    reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
// transpose A during the GMEM to SMEM transfer
As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
    reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
__syncthreads();
```

This leads to the 32b GMEM load instructions (`LDG.E` and `STG.E`) being replaced with 128b counterparts (`LDG.E.128` and `STG.E.128`). Initially, I was confused as to why running this:

````c++
reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
    reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];```

would be any faster than just manually unrolling the access (or using `pragma unroll`):

```c++
Bs[innerRowB * BN + innerColB * 4 + 0] = B[innerRowB * N + innerColB * 4 + 0];
Bs[innerRowB * BN + innerColB * 4 + 1] = B[innerRowB * N + innerColB * 4 + 1];
Bs[innerRowB * BN + innerColB * 4 + 2] = B[innerRowB * N + innerColB * 4 + 2];
Bs[innerRowB * BN + innerColB * 4 + 3] = B[innerRowB * N + innerColB * 4 + 3];
````

Shouldn’t the compiler just be able to coalesce the 2nd version and also generate 128b loads? I think the reason is that the compiler has no way to verify that the `float* B` pointer that is passed to the kernel is 128b aligned, which would be a requirement for using `LDG.E.128`. So the `reinterpret_cast`’s only purpose is to promise the compiler that the `float* B` pointer will be aligned.

Kernel 6 achieves 19TFLOPs. The profiler still shows a bunch of problem areas and optimization opportunities: We’re running into shared-memory bank conflicts (which cuBLAS avoids), our occupancy is higher than necessary, and we haven’t implemented any double buffering (which the [CUTLASS docs](https://github.com/NVIDIA/cutlass/blob/master/media/docs/efficient_gemm.md#pipelining) seem to suggest is pretty useful).

But before we get to those, let’s cover some more low-hanging fruit: Autotuning the kernel’s parameters.

## Kernel 9: Autotuning

We’ve accumulated a total of five template parameters:

- `BM`, `BN` and `BK`, which specify how much data we cache from GMEM into SMEM.
- `TM` and `TN`, which specify how much data we cache from SMEM into the registers.

For kernel 6, these were set to `BM=BN=128` and `BK=TM=TN=8`. I wrote a bash script that searches through all sensible combinations and benchmarks their runtime. This required me to make sure that:

1. I knew which parameter combinations were sensible, and skip those that weren’t.
2. The kernel implementation was correct for the ~400 different hyperparameter settings that remained.

The necessary modifications to the code ended up taking quite some time to implement.

It turns out that the optimal parameters vary quite a bit depending on the GPU model. On my A6000, `BM=BN=128 BK=16 TM=TN=8` increased performance by 5%, from 19 to 20 TFLOPs. On an A100 SMX4 40GB, that same configuration reached 12 TFLOPs, 6% worse than the optimal setting found by the autotuner (`BM=BN=64 BK=16 TM=TN=4`), which reached 12.6 TFLOPs.

I can’t explain why these specific parameters end up producing the optimal performance. Autotuning works, every high-performance library uses it, but it also feels very unsatisfying.

## Kernel 10: Warptiling

Currently, our loop structure looks like this:

![Loop_structure](/images/CUDA-MMM/Loop_structure.png)

We’ll now add another hierarchy of tiling, in between our blocktiling and threadtiling loops: warptiling. Warptiling is somewhat confusing initially since unlike blocks and threads, warps don’t show up anywhere in the CUDA code explicitly. They are a hardware feature that has no direct analog in the scalar CUDA-software world. We can calculate a given thread’s warpId as `warpId=threadIdx.x % warpSize`, where `warpSize` is a built-in variable that is equal to 32 on any CUDA GPU I’ve ever worked with.

Warps are relevant for performance since (among other reasons):

- Warps are the unit of scheduling that is mapped to the warp-schedulers that are part of the SM.
- Shared-memory bank conflicts (I’ll cover those in a future post) happen only between threads that are in the same warp.
- There’s a register cache on recent GPUs, and tighter threadtiling gives us more register cache locality.

Warptiling is elegant since we now make explicit all levels of parallelism:

- Blocktiling: Different blocks can execute in parallel on different SMs.
- Warptiling: Different warps can execute in parallel on different warp schedulers, and concurrently on the same warp scheduler.
- Threadtiling: (a very limited amount of) instructions can execute in parallel on the same CUDA cores (= instruction-level parallelism aka ILP).

The warptiling looks like this in the CUDA code:

```cpp
// dotIdx loops over contents of SMEM
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
  // populate registers for this thread's part of the warptile
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint i = 0; i < TM; ++i) {
      regM[wSubRowIdx * TM + i] =
          As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
             threadRowInWarp * TM + i];
    }
  }
  for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
    for (uint i = 0; i < TN; ++i) {
      regN[wSubColIdx * TN + i] =
          Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
             threadColInWarp * TN + i];
    }
  }

  // execute warptile matmul. Later this will map well to
  // warp-wide matrix instructions, executed on tensor cores.
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // calculate per-thread results with register-cache locality
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        (wSubColIdx * TN) + resIdxN] +=
              regM[wSubRowIdx * TM + resIdxM] *
              regN[wSubColIdx * TN + resIdxN];
        }
      }
    }
  }
}
```

I tried my best to visualize all three levels of tiling below, although the structure is getting quite complex. Each warp will compute a chunk of size `(WSUBN * WNITER) x (WSUBM * WMITER)`. Each thread computes `WNITER * WMITER` many chunks of size `TM*TN`.

![kernel_10_warp_tiling](/images/CUDA-MMM/kernel_10_warp_tiling.png)

After autotuning the parameters, performance improves from 19.7 TFLOPs to 21.7 TFLOPs on an A100.

Here’s a plot that compares our warptiling kernel against cuBLAS across increasing matrix sizes:

![cublas_vs_kernel_10_sizes](/images/CUDA-MMM/cublas_vs_kernel_10_sizes.png)

At dimensions 2048 and 4096, our measured FLOPs are only a few percentage points slower than cuBLAS. However, for smaller matrices, we’re doing poorly in comparison to Nvidia’s library! This happens because cuBLAS contains not one single implementation of SGEMM, but hundreds of them. At runtime, based on the dimensions, cuBLAS will pick which kernel to run. I traced the cuBLAS call and these are the kernels it’s calling at each size:

| Matrix size | Name                                                                  | Duration             |
| :---------- | :-------------------------------------------------------------------- | :------------------- |
| 128         | `ampere_sgemm_32x32_sliced1x4_nn`                                     | 15.295 μs            |
| 256         | `ampere_sgemm_64x32_sliced1x4_nn` _followed by_ `splitKreduce_kernel` | 12.416 μs + 6.912 μs |
| 512         | `ampere_sgemm_32x32_sliced1x4_nn`                                     | 41.728 μs            |
| 1024        | `ampere_sgemm_128x64_nn`                                              | 165.953 μs           |
| 2048        | `ampere_sgemm_128x64_nn`                                              | 1.247 ms             |
| 4096        | `ampere_sgemm_128x64_nn`                                              | 9.290 ms             |

At dimension 256 it calls two kernels: a matmul kernel followed by a reduction kernel. So if we were trying to write a high-performance library that works for all shapes and sizes we would have specializations for different shapes, and at runtime dispatch to the one that’s the best fit.

I also want to report a negative results: For this kernel, I additionally implemented an optimization called _thread swizzling_. This technique assumes that threadblocks are launched in order of increasing `blockIdx`, and optimizes the mapping of `blockIdx` to C chunks in a way that should increase L2 locality. This [Nvidia post](https://developer.nvidia.com/blog/optimizing-compute-shaders-for-l2-locality-using-thread-group-id-swizzling/) has more info and visualizations. It didn’t increase performance, presumably because L2 hit rate is already fairly high at 80%, so I ended up removing the swizzling code.

It makes sense to move the loop over BK towards the outside, since it follows our maxim of “load some data, then do as much work on that data as possible”. It further means that all _computation_ that happens inside the BK loop will be independent and can be parallelized (for example using ILP).

We can now also start prefetching the data necessary for the next loop iteration already, a technique called double buffering.

## Work in Progress: Kernel 11

If I get back to working on this post, here’s what I’ll look at next:

1. Double buffering, for better interleaving of computation and memory loading. For now, see [CUTLASS Pipelining](https://github.com/NVIDIA/cutlass/blob/master/media/docs/efficient_gemm.md#pipelining). In CUTLASS, double buffering is done on two levels: GMEM ⇒ SMEM, and SMEM ⇒ Registerfile.
    - In Hopper, new instructions were introduced for warp specialization, for example for having some warp use fewer registers than others. This, in combination with special instructions to load directly from GMEM into SMEM without first going through the registers, can be used to reduce register pressure.
2. Getting rid of SMEM bank conflicts. This can be done by [optimizing the data layout in SMEM](https://github.com/NVIDIA/cutlass/blob/master/media/docs/implicit_gemm_convolution.md#shared-memory-layouts).
3. Better understanding the GEMM kernels that are implemented in [Triton](https://github.com/openai/triton), by looking at the generated PTX.

## Conclusion

Writing this post was a similar experience to my previous post on [optimizing SGEMM on CPU](/articles/22/Fast-MMM-on-CPU): Optimizing SGEMM iteratively is one of the best ways to deeply understand the performance characteristics of the hardware. For writing the CUDA programs I was surprised by how easy it was to implement the code once I had made a good visualization of how I wanted the kernel to work.

Also: Powerlaws are everywhere. It took me two weekends to write the first 6 kernels which reach 80% of peak FLOPs, and then 4 more weekends to do autotuning and warptiling to get to 94%. How much I’m learning while writing this code has also seen diminishing results, hence I’m putting off hunting the last 6% until some future time.

All my code is available on [Github](https://github.com/siboehm/SGEMM_CUDA).

Lastly, a big thanks to the creators of [Godbolt.org](https://godbolt.org/) (for looking at PTX and SASS assembly) and [Excalidraw](https://excalidraw.com/) (for drawing the kernels)! Both of these tools are a joy to use and have helped me learn much faster.

## Further Resources and References

- I started writing this post because I stumbled over [wangzyon’s Github repository](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE), first experimenting with his kernels and then rewriting everything from scratch. Also relevant is this [Nvidia Blogpost about the CUTLASS library](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/).
- Mandatory references: the official [CUDA Toolkit Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) and the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide). The [Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) contains even more info on low-level hardware details like caches and pipelines, and on the various metrics that can be collected.
- Onur Mutlu is a professor at ETH who uploads his lectures to Youtube. Particularly relevant for this post are [Computer Architecture](https://www.youtube.com/playlist?list=PL5Q2soXY2Zi-Mnk1PxjEIG32HAGILkTOF) and [Acceleration on Heterogeneuous Systems](https://www.youtube.com/playlist?list=PL5Q2soXY2Zi_OwkTgEyA6tk3UsoPBH737).
- [Understanding Latency Hiding on GPUs](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.pdf), a Ph.D. thesis that goes in-depth on how to design workloads such that they fully utilize memory bandwidth and computation. It’s from 2016 and hence only covers older GPU architectures. The chapter about warp-synchronous programming is outdated, see [using CUDA warp-level primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/).
- Lei Mao (an engineer at Nvidia) has good CUDA content on his [blog](https://leimao.github.io/tags/CUDA/), including about [proper CUDA error handling](https://leimao.github.io/blog/Proper-CUDA-Error-Checking/).
- It seems like there aren’t any good official resources for understanding SASS. There is [Nvidia’s Docs on CUDA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html). More useful might be looking at Open Source SASS assemblers, like Da Yan’s [turingas](https://github.com/daadaada/turingas).
- I’m collecting examples of readable, yet optimized CUDA code to learn from:
  - ONNX Runtime’s [CUDA provider](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/cuda), e.g. their [implementation of softmax](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cuda/math/softmax_warpwise_impl.cuh).
  - NVIDIA’s Open Source [CUTLASS](https://github.com/NVIDIA/cutlass) library, e.g. their [GEMM implementation](https://github.dev/NVIDIA/cutlass/blob/master/include/cutlass/gemm/device/gemm.h) which uses double-buffering to prefetch the innermost dimension, which is still missing in my kernels.
