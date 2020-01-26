

// general
#include <stdio.h>
#include <conio.h>
#include <iostream>

// cuda
#include "cuda_runtime.h"


int main()
{

	int deviceNum, deviceId;
	cudaGetDeviceCount(&deviceNum);
	printf("Available devices = %d pc(s)\n\n", deviceNum);

	cudaDeviceProp prop;
	for (deviceId = 0; deviceId < deviceNum; deviceId++)
	{
		cudaGetDeviceProperties(&prop, deviceId);

		// print selected device capabilities
		printf("Device properties:\n\n");
		printf("    Device id                                : %d\n", deviceId);
		printf("    Device name                              : %s\n", prop.name);
		printf("    Compute capability                       : %d.%d\n", prop.major, prop.minor);
		printf("    Total global mem                         : %llu MB\n", prop.totalGlobalMem / 1024 / 1024);
		printf("    Number of SMs                            : %d\n", prop.multiProcessorCount);
		printf("\n");
		printf("    Registers & memory \n");
		printf("        available registers per SM           : %lu k\n", prop.regsPerMultiprocessor / 1024);
		printf("        available shared mem per SM          : %llu kB\n", prop.sharedMemPerBlock / 1024);
		printf("        L2 cache size per SM                 : %d kB\n", prop.l2CacheSize / 1024);
		printf("\n");
		printf("    Thread and block data\n");
		printf("        threads per warp                     : %d\n", prop.warpSize);
		printf("        maximum threads per block            : %d\n", prop.maxThreadsPerBlock);
		printf("        maximum threads per SM               : %d\n", prop.maxThreadsPerMultiProcessor);
		printf("\n");

		// calculate number of blocks and threads
		dim3 threads(prop.warpSize, prop.maxThreadsPerBlock / prop.warpSize); // maximum number of threads per block in wapr groups
		dim3 blocks(prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / prop.maxThreadsPerBlock, 1, 1); // all block will be active! - mutex is possible

		#define threadsPerBlock (threads.x * threads.y * threads.z)
		#define totalBlocks (blocks.x * blocks.y * blocks.z)
		#define activeBlocksPerSM  (prop.maxThreadsPerMultiProcessor / threadsPerBlock)

		printf("\n");
		printf("    Settings for full occupancy and maximum sawpping  :\n\n");
		printf("\n");
		printf("        Kernel config:\n");
		printf("            total number of blocks           : %d x %d x %d = %d\n", blocks.x, blocks.y, blocks.z, totalBlocks);
		printf("            block population                 : %d cycles x %d blocks + %d blocks\n", totalBlocks / activeBlocksPerSM / prop.multiProcessorCount, activeBlocksPerSM * prop.multiProcessorCount, totalBlocks % (activeBlocksPerSM * prop.multiProcessorCount));
		printf("            SM occupancy                     : %d \n", 100 * totalBlocks / activeBlocksPerSM / prop.multiProcessorCount);
		printf("\n");
		printf("        Block data:\n");
		printf("            threads per block                : %d x %d x %d = %d\n", threads.x, threads.y, threads.z, threadsPerBlock);
		printf("            warps per block                  : %d\n", threadsPerBlock / prop.warpSize);
		printf("            maximum shared mem per block     : %llu kB\n", prop.sharedMemPerBlock / activeBlocksPerSM / 1024);
		printf("\n");
		printf("        Thread data:\n");
		printf("            average registers per thread     : %d\n", prop.regsPerBlock / activeBlocksPerSM / threadsPerBlock);
		printf("            average shared mem per thread    : %llu B\n", prop.sharedMemPerBlock / activeBlocksPerSM / threadsPerBlock);
		printf("\n");
		printf("        Others:\n");
		printf("            L1 cache transaction size        : %d B\n", 128);
		printf("            L2 cache transaction size        : %d B (only L2 = uncached loads using the generic data path)\n", 32);
		printf("            texture cache transaction size   : %d B\n", 32);
		if (prop.major <= 2) printf("            L1 cache is separated from shared mem! Use cudaDeviceSetCacheConfig() !\n");
		if (prop.major >= 3) printf("            Shuffle available!\n");
		printf("\n");

	}
}

