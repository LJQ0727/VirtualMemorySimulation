run: main
	./main
main: main.cu virtual_memory.cu user_program.cu
	nvcc --relocatable-device-code=true main.cu user_program.cu virtual_memory.cu -o main
