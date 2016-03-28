Hong Joon Park


PART 1 (Time taken: 5 hrs 15 mins)
Question 1.1: Latency Hiding
---------------------------------------
The latency of an arithmetic instruction is 10 ns. The number of instructions per cycle for a GK110 is 8 (4 warps each clock, up to 2 instructions in each warp). A GPU clock is 1 GHz (1 clock per ns). Thus, for 1 ns, the GPU undergoes 1 cycle, and 8 instructions in this cycle.

Since an arithmetic instruction takes 10 ns, 10 * 8 = 80 instructions can take place in the span of an arithmetic instruction. Thus, it takes 80 instructions to hide the latency of a single arithmetic instruction.


Question 1.2: Thread Divergence
------------------------------------------
Let the block shape be (32, 32, 1).

(a)
This code does not diverge. Given our block shape, we see that the warps are organized in such a way that each individual threadIdx.y has its own warp of 32 threads, comprising of the threadIdx.x’s. Looking at our idx formula, we see that since our blockSize.y = 32, the idx depends solely on threadIdx.y (since we are applying a mod 32 on idx). We’ve already established that each warp has its own unique threadIdx.y, thus this code does not diverge.

(b)
This code does diverge. For each threadIdx.x, that thread calculates the for-loop while the rest of the threads in that warp halt (since the warps comprise of the threads of threadIdx.x for a single threadIdx.y).


Question 1.3: Coalesced Memory Access
------------------------------------------------
Let the block shape be (32, 32, 1).
Let data be a (float *) pointing to global memory and let data be 128 byte
aligned (so data % 128 == 0).

Consider each of the following access patterns.

(a)
This write is coalesced. For a single warp, this writes to one 128-byte cache line.


(b)
This write is not coalesced. For a single warp, this writes to 32 128-byte cache lines.


(c)
This write is not coalesced. For a single warp, this writes to two 128-byte cache lines.


Question 1.4: Bank Conflicts and Instruction Dependencies
---------------------------------------------------------------------
Let's consider multiplying a 32 x 128 matrix with a 128 x 32
element matrix. This outputs a 32 x 32 matrix. We'll use 32 ** 2 = 1024 threads
and each thread will compute 1 output element.
Although its not optimal, for the sake of simplicity let's use a single block,
so grid shape = (1, 1, 1), block shape = (32, 32, 1).

For the sake of this problem, let's assume both the left and right matrices have
already been stored in shared memory are in column major format. This means
element in the ith row and jth column is accessible at lhs[i + 32 * j] for the
left hand side and rhs[i + 128 * j] for the right hand side.

This kernel will write to a variable called output stored in shared memory.

Consider the following kernel code:

int i = threadIdx.x;
int j = threadIdx.y;
for (int k = 0; k < 128; k += 2) {
    output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
    output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
}

(a)
There are no bank conflicts in this code.


(b)
lhs0 = lhs[i + 32 * k];
rhs0 = rhs[k + 128 * j];
O0 = output[i + 32 * j];
FMA on lhs0, rhs0, O0;
Write O0 to output[i + 32 * j];
lhs1 = lhs[i + 32 * (k + 1)];
rhs1 = rhs[(k + 1) + 128 * j];
O1 = output[i + 32 * j];
FMA on lhs1, rhs1, O1;
Write O1 to output[i + 32 * j];


(c)
“Write O0 to output[i + 32 * j];” depends on “FMA on lhs0, rhs0, O0;”. “Write O1 to output[i + 32 * j];” depends on “FMA on lhs1, rhs1, O1;”. “FMA on lhs0, rhs0, O0;” depends on “lhs0, rhs0, and O0”. “FMA on lhs1, rhs1, O1;” depends on “lhs1, rhs1, and O1”. 


(d)
lhs0 = lhs[i + 32 * k];
rhs0 = rhs[k + 128 * j];
lhs1 = lhs[i + 32 * (k + 1)];
rhs1 = rhs[(k + 1) + 128 * j];
O = output[i + 32 * j];
FMA on lhs0, rhs0, O;
FMA on lhs1, rhs1, O;
Write O to output[i + 32 * j];

(e)
Why stop at two values of k? Let’s repeat (d), but use more values of k (say, processing 4 values of k rather than 2 by doing k, (k + 1), (k + 2), (k + 3)). 


================================================================================

PART 2 - Matrix transpose optimization (Time taken: 13 hrs)

hppark@haru:~/Problem Set 2$ ./transpose 
Size 512 naive CPU: 0.299488 ms
Size 512 GPU memcpy: 0.032000 ms
Size 512 naive GPU: 0.093216 ms
Size 512 shmem GPU: 0.030304 ms
Size 512 optimal GPU: 0.029056 ms

Size 1024 naive CPU: 2.188064 ms
Size 1024 GPU memcpy: 0.081856 ms
Size 1024 naive GPU: 0.314496 ms
Size 1024 shmem GPU: 0.094080 ms
Size 1024 optimal GPU: 0.089152 ms

Size 2048 naive CPU: 37.497826 ms
Size 2048 GPU memcpy: 0.266592 ms
Size 2048 naive GPU: 1.169344 ms
Size 2048 shmem GPU: 0.346592 ms
Size 2048 optimal GPU: 0.324032 ms

Size 4096 naive CPU: 156.696838 ms
Size 4096 GPU memcpy: 0.997888 ms
Size 4096 naive GPU: 4.142496 ms
Size 4096 shmem GPU: 1.262656 ms
Size 4096 optimal GPU: 1.254368 ms


================================================================================
