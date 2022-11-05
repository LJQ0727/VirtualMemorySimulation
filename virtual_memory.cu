#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __managed__ u32 access_trace = 0;  // For LRU implementation.

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
  }
}

__device__ void init_swap_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->SWAP_PAGE_ENTRIES; i++) {
    vm->swap_table[i] = 0x80000000; // invalid := MSB is 1
  }
}

// Initialize the attributes of the VirtualMemory struct
// Initialize the page table by calling `init_invert_page_table`
__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES, u32 *swap_table) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  vm->swap_table = swap_table;
  vm->SWAP_PAGE_ENTRIES = STORAGE_SIZE / PAGESIZE;


  // before first vm_write or vm_read
  init_invert_page_table(vm);
  init_swap_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  access_trace++;

  return 123; //TODO
}

__device__ int my_log2(int num) {
  // get the log of 2^n, returning n
  int ret = 0;
  while (num != 1)
  {
    num = num >> 1;
    ret++;
  }
  return ret;
}

__device__ u32 alloc_page() {
  // allocate a page from the physical memory
  // return the page number

}

__device__ u32 swap_page() {

}

// Use the LRU algorithm to find a pointer to an entry in the page table that is least recently used
__device__ int LRU_get(VirtualMemory *vm) {
  int swapped_frame_no = 0;
  u32 least_trace_value = vm->invert_page_table[0+vm->PAGE_ENTRIES];

  for (int i = 1; i < vm->PAGE_ENTRIES; i++)
  {
    if (vm->invert_page_table[i+vm->PAGE_ENTRIES] < least_trace_value)
    {
      swapped_frame_no = i;
      least_trace_value = vm->invert_page_table[i+vm->PAGE_ENTRIES];
    }
    
  }
  return swapped_frame_no;
}

__device__ void LRU_put(VirtualMemory *vm, int frame_no) {
  vm->invert_page_table[frame_no+vm->PAGE_ENTRIES] = access_trace;
}

__device__ void swap(VirtualMemory *vm, int swapped_frame_no, int storage_frame_no) {
  // TODO
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  access_trace++;

  // given 32-bit virtual address addr, compute the page number and offset fields
  int offset_bit = my_log2(vm->PAGESIZE); // 5-bit offset in each frame (or page)
  int page_entries_bit = my_log2(vm->PAGE_ENTRIES); // 10-bit page entry

  int page_number = addr >> offset_bit; // This page number has at most 13 bits for our problem
  int offset = addr & ((1 << offset_bit) - 1);

  bool page_is_found = false;
  // in the inverted page table, search for the page number
  // i is the frame number
  for (int i = 0; i < vm->PAGE_ENTRIES; i++)
  {
    // if page is found in the page table
    u32 entry = vm->invert_page_table[i];
    if ((entry & 0x80000000 == 0) && entry & 0x7FFFFFFF == page_number) // if the page number is found
    {
      page_is_found = true;
      // get the frame number
      int frame_number = i;
      // write the value into the buffer
      vm->buffer[frame_number * vm->PAGESIZE + offset] = value;
      LRU_put(vm, frame_number);
      return;
    } 
  } 
  
  if (!page_is_found) {
    // if page is not found in the page table
    vm->pagefault_num_ptr[0]++;

    // check if the primary memory is full
    // if the primary memory is not full, we can directly allocate new page there
    for (int i = 0; i < vm->PAGE_ENTRIES; i++)
    {
      if (vm->invert_page_table[i] & 0x80000000 == 1) {
        // this entry is not used
        // mark it occupied
        vm->invert_page_table[i] = 0;

        // write the page number
        vm->invert_page_table[i] &= page_number;

        // write to destination
        vm->buffer[i + offset] = value;

        vm->invert_page_table[i+vm->PAGE_ENTRIES] = access_trace;
        
        return;
      }
    }
    
    // the main memory is full, we have to swap in an exsting page or swap in a new page

    // first, we determine whether the page just doesn't exist yet.
    // find in the swap table
    for (int i = 0; i < vm->SWAP_PAGE_ENTRIES; i++)
    {
      u32 entry = vm->swap_table[i];
      if ((entry & 0x80000000 == 0) && entry & 0x7FFFFFFF == page_number) // if the page number is found
      {
        page_is_found = true;
        
        // the page in found in the swap storage
        // now we swap with an LRU in the main memory
        int swapped_frame_no = LRU_get(vm);

        // do the swapping

        // write the value into the buffer
        vm->buffer[frame_number * vm->PAGESIZE + offset] = value;
        return;
      } 
    }
    
    




    // if not, allocate a page from the physical memory
    int frame_number = alloc_page();
    // write the value into the buffer
    vm->buffer[frame_number * vm->PAGESIZE + offset] = value;
    // update the page table
    vm->invert_page_table[page_number] = frame_number;
    // update the page fault number
    return;
  }
  
  // map to physical address given the page, do the write, set the valid bit


  u32 frame_number = vm->invert_page_table[page_number]; // each page entry is 32-bit
  if (frame_number & 0x80000000 > 0) {
    // allocate a page from the physical memory, or if it's full we'll have to swap
    // The last ten bits of the physical address will be
    frame_number = alloc_page();

    // unset the MSB, indicating valid address

    frame_number = frame_number & 0x7FFFFFFF;
    

  }


  // write to the physical address


}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
}

