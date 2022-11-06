#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cassert>

__device__ __managed__ u32 access_trace = 0;  // For LRU implementation.

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = 0;
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
  // swap the frame in the main memory with the frame in the physical memory
  // also swap the page table entries
  u32 temp = vm->invert_page_table[swapped_frame_no];
  vm->invert_page_table[swapped_frame_no] = vm->swap_table[storage_frame_no];
  vm->swap_table[storage_frame_no] = temp;
  for (int i = 0; i < vm->PAGESIZE; i++)
  {
    u32 tmp = vm->buffer[swapped_frame_no*vm->PAGESIZE + i];
    vm->buffer[swapped_frame_no*vm->PAGESIZE + i] = vm->storage[storage_frame_no*vm->PAGESIZE + i];
    vm->storage[storage_frame_no*vm->PAGESIZE + i] = tmp;
  }
  
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  access_trace++;

  // given 32-bit virtual address addr, compute the page number and offset fields
  int offset_bit = my_log2(vm->PAGESIZE); // 5-bit offset in each frame (or page)
  int page_entries_bit = my_log2(vm->PAGE_ENTRIES); // 10-bit page entry

  int page_number = addr >> offset_bit; // This page number has at most 13 bits for our problem
  int offset = addr & ((1 << offset_bit) - 1);


  for (int i = 0; i < vm->PAGE_ENTRIES; i++)
  {
    u32 entry = vm->invert_page_table[i];
    if (entry == page_number)
    {
      // page is found in the main memory
      LRU_put(vm, i);
      // printf("value is %d\n", vm->buffer[i*vm->PAGESIZE + offset]);
      return vm->buffer[i*vm->PAGESIZE + offset];
    }
  }
    // page is not found in the main memory
    // this is a page fault
    // now we have to find in the swap table
    (*vm->pagefault_num_ptr)++;
    for (int j = 0; j < vm->SWAP_PAGE_ENTRIES; j++)
    {
      u32 entry = vm->swap_table[j];
      if (entry == page_number)
      {
        // page is found in the swap table
        // we have to swap the page in the main memory with the page in the swap table
        int swapped_frame_no = LRU_get(vm);
        swap(vm, swapped_frame_no, j);
        LRU_put(vm, swapped_frame_no);
        return vm->buffer[swapped_frame_no*vm->PAGESIZE + offset];
      }
    }
    assert(0);  // no such page.
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
    if (entry == page_number) // if the page number is found
    {
      page_is_found = true;
      // get the frame number
      int frame_number = i;
      // printf("frame number is %d, offset %d\n", frame_number, offset);
      // write the value into the buffer
      vm->buffer[frame_number * vm->PAGESIZE + offset] = value;
      LRU_put(vm, frame_number);
      return;
    } 
  } 


  if (!page_is_found) {
    // if page is not found in the page table
    (*vm->pagefault_num_ptr)++;


    // check if the main memory is full
    // if the main memory is not full, we can directly allocate new page there
    for (int i = 0; i < vm->PAGE_ENTRIES; i++)
    {
      if ((vm->invert_page_table[i] & 0x80000000) != 0) {
        // this entry is not used
        // mark it occupied and
        // write the page number
        vm->invert_page_table[i] = page_number;

        // write to destination
        vm->buffer[i * vm->PAGESIZE + offset] = value;

        LRU_put(vm, i);
        
        return;
      }
    }

    // the main memory is full, we have to swap in an exsting page or swap in a new page

    // first, we determine whether the page just doesn't exist yet.
    // find in the swap table
    for (int i = 0; i < vm->SWAP_PAGE_ENTRIES; i++)
    {
      u32 entry = vm->swap_table[i];
      if (entry == page_number) // if the page number is found
      {
        //printf("page is found in the swap table");
        // the page in found in the swap storage
        page_is_found = true;
        
        // now we swap with an LRU in the main memory
        int swapped_frame_no = LRU_get(vm);



        // do the swapping
        swap(vm, swapped_frame_no, i);

        // write the value into the buffer
        vm->buffer[swapped_frame_no * vm->PAGESIZE + offset] = value;

        LRU_put(vm, swapped_frame_no);
        return;
      } 
    }
    
    // the page is not found in the swap storage, meaning it doesn't exist yet
    // allocate an unoccupied page from the physical memory
    for (int i = 0; i < vm->SWAP_PAGE_ENTRIES; i++)
    {
      u32 entry = vm->swap_table[i];
      if ((entry & 0x80000000) != 0) {
        // this entry is not used
        // mark it occupied and
        // write the page number
        vm->swap_table[i] = page_number;
        page_is_found = true;

        // do the swapping
        int swapped_frame_no = LRU_get(vm);

        // printf("swapped frame no: %d\n", swapped_frame_no);


        swap(vm, swapped_frame_no, i);

        // write to destination
        vm->buffer[swapped_frame_no * vm->PAGESIZE + offset] = value;

        LRU_put(vm, swapped_frame_no);
        return;
      }
    }
  }
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  u32 addr = offset;
  for (int i = 0; i < input_size; i++) {
    results[i] = vm_read(vm, addr);
    addr++;
  }
}

