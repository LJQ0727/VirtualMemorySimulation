#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

struct VirtualMemory {
  uchar *buffer;  // pointer to shared memory, 32kb data
  uchar *storage; // pointer to the 128kb static memory
  u32 *invert_page_table;   // pointer to share dynamic memory, page table
  
  u32 *swap_table;
  int *pagefault_num_ptr;

  int PAGESIZE; // set to 32. 32 bytes
  int INVERT_PAGE_TABLE_SIZE; // size is 16kb
  int PHYSICAL_MEM_SIZE;  // 32kb in shared mem
  int STORAGE_SIZE; // 128kb in global mem
  int PAGE_ENTRIES;   // number of page entries, here it's PHYSICAL_MEM_SIZE / PAGE_SIZE, 1024

  int SWAP_PAGE_ENTRIES;  // number of page entries in swap table. which is 4096
  int thread_id;  // thread id, 0, 1, 2, 3
};

// TODO
__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES, u32 *swap_table);
__device__ uchar vm_read(VirtualMemory *vm, u32 addr);
__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value);
__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size);

#endif
