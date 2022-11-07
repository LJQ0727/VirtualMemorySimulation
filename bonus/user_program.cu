#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
  for (int i = 0; i < input_size; i++)
    vm_write(vm, i, input[i]);
  // this will produce 16384 page faults

  // for (int i = input_size - 1; i >= input_size - 8192; i--)
  //   int value = vm_read(vm, i);
  // this is just for testing. this will produce 0 additional page faults, becuase we're reading everything in main memory.

  for (int i = input_size - 1; i >= input_size - 32769; i--)
    int value = vm_read(vm, i);
  // this will produce 19460 - 16384 = 3076 page faults
  

  vm_snapshot(vm, results, 0, input_size);
  // the test1 program in total will produce 35844 page faults in my implementation of Case3.
}

// __device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
//   int input_size) {
// // write the data.bin to the VM starting from address 32*1024
// for (int i = 0; i < input_size; i++)
//   vm_write(vm, 32*1024+i, input[i]);
// // write (32KB-32B) data  to the VM starting from 0
// for (int i = 0; i < 32*1023; i++)
//   vm_write(vm, i, input[i+32*1024]);
// // readout VM[32K, 160K] and output to snapshot.bin, which should be the same with data.bin
// vm_snapshot(vm, results, 32*1024, input_size);
// }

// expected page fault num: 9215