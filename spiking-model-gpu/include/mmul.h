// Pull out matrix and shared memory tile size 
const int N = 1 << 10;
const int SHMEM_SIZE = 1 << 10;

void matrixMultiply(const int *a, const int *b, int *c);
