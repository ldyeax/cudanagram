// gpu_caps.cu
// Build: nvcc -O2 -std=c++17 gpu_caps.cu -o gpu_caps
// Usage: ./gpu_caps [block_size] [problem_size]
//   block_size   : optional, default 256
//   problem_size : optional, default 0 (skip per-N launch math)
// Notes:
// - "CUDA cores" (SPs) per SM are architecture-dependent and not an official API.
//   The table below is a best-effort mapping for common architectures.

//##include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

static int coresPerSM(int major, int minor) {
    // Best-effort mapping. NVIDIA does not expose this via the runtime API.
    // References: public whitepapers and community docs (may vary by chip variant).
    // Kepler (3.x): 192
    // Maxwell (5.x): 128
    // Pascal (6.0 GP100: 64; 6.1/6.2 GP10x: 128)
    // Volta (7.0): 64
    // Turing (7.5): 64
    // Ampere (8.0 GA100: 64; 8.6 GA10x: 128)
    // Ada Lovelace (8.9): 128
    // Hopper (9.0): 128 (approx.)
    // Fallback: 64
    if (major == 3) return 192;              // Kepler
    if (major == 5) return 128;              // Maxwell
    if (major == 6) {                         // Pascal
        if (minor == 0) return 64;           // GP100
        return 128;                          // GP10x
    }
    if (major == 7) {                         // Volta/Turing
        return 64;
    }
    if (major == 8) {                         // Ampere/Ada
        if (minor == 0) return 64;           // GA100
        if (minor == 6) return 128;          // GA10x
        if (minor == 9) return 128;          // Ada (approx.)
        return 64;                           // conservative
    }
    if (major >= 9) {                         // Hopper and beyond (approx.)
        return 128;
    }
    // Fermi etc.
    return 64;
}

// A trivial kernel for occupancy demonstration (minimal registers/shared mem).
__global__ void dummy_kernel() { /* no-op */ }

static std::string bytesNice(size_t b) {
    const char* units[] = {"B","KB","MB","GB","TB"};
    double val = static_cast<double>(b);
    int u = 0;
    while (val >= 1024.0 && u < 4) { val /= 1024.0; ++u; }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(val < 10 ? 2 : (val < 100 ? 1 : 0)) << val << " " << units[u];
    return oss.str();
}

int main(int argc, char** argv) {
    int blockSize = 256;
    long long problemSize = 0;
    if (argc >= 2) blockSize = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) problemSize = std::atoll(argv[2]);

    int deviceCount = 0;
    cudaError_t st = cudaGetDeviceCount(&deviceCount);
    if (st != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(st) << "\n";
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found.\n";
        return 0;
    }

    std::cerr << "Found " << deviceCount << " CUDA device(s)\n\n";

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp p{};
        cudaGetDeviceProperties(&p, dev);

        // Derived numbers
        const int smCount = p.multiProcessorCount;
        const int spPerSM = coresPerSM(p.major, p.minor);
        const long long estSPs = 1LL * smCount * spPerSM;

        // Memory bandwidth (approx): memClockRate is kHz, bus width in bits.
        // DDR => effective rate is typically 2x; for GDDR6/GDDR6X the effective multiplier is higher,
        // but we’ll conservatively use 2x here to avoid overstatement.
        double memClockMHz = p.memoryClockRate / 1000.0; // kHz -> MHz
        double busWidthBytes = p.memoryBusWidth / 8.0;   // bits -> bytes
        double approxBW_GBps = (memClockMHz * 1e6) * busWidthBytes * 2.0 / 1e9; // ~GB/s

        std::cerr << "Device " << dev << " : " << p.name << "\n";
        std::cerr << "  Compute Capability        : " << p.major << "." << p.minor << "\n";
        std::cerr << "  SMs (Streaming MPs)       : " << smCount << "\n";
        std::cerr << "  Est. CUDA cores (SPs)     : " << estSPs
                  << "  (" << spPerSM << " per SM; approx)\n";
        std::cerr << "  Global Memory             : " << bytesNice(p.totalGlobalMem) << "\n";
        if (p.l2CacheSize > 0) {
            std::cerr << "  L2 Cache                  : " << bytesNice(p.l2CacheSize) << "\n";
        }
        std::cerr << "  Shared mem per block      : " << bytesNice(p.sharedMemPerBlock) << "\n";
#if CUDART_VERSION >= 11000
        std::cerr << "  Shared mem per SM (opt)   : " << bytesNice(p.sharedMemPerMultiprocessor) << "\n";
#endif
        std::cerr << "  Registers per block       : " << p.regsPerBlock << "\n";
        std::cerr << "  Warp size                 : " << p.warpSize << "\n";
        std::cerr << "  Max threads / block       : " << p.maxThreadsPerBlock << "\n";
        std::cerr << "  Max threads / SM          : " << p.maxThreadsPerMultiProcessor << "\n";
        std::cerr << "  Max threads dims          : [" << p.maxThreadsDim[0] << ", "
                                                   << p.maxThreadsDim[1] << ", "
                                                   << p.maxThreadsDim[2] << "]\n";
        std::cerr << "  Max grid dims             : [" << p.maxGridSize[0] << ", "
                                                   << p.maxGridSize[1] << ", "
                                                   << p.maxGridSize[2] << "]\n";
        std::cerr << "  Clock rate (core)         : " << (p.clockRate/1000.0) << " MHz\n";
        std::cerr << "  Mem clock / bus           : " << memClockMHz << " MHz / " << p.memoryBusWidth << " bits\n";
        std::cerr << "  Approx mem bandwidth      : " << std::fixed << std::setprecision(1)
                  << approxBW_GBps << " GB/s (conservative)\n";
        std::cerr << "  PCI Bus:Device:Domain     : " << p.pciBusID << ":" << p.pciDeviceID
#if CUDART_VERSION >= 10000
                  << " (domain " << p.pciDomainID << ")\n";
#else
                  << "\n";
#endif
        std::cerr << "  Unified Addr / ECC / ConK : "
                  << (p.unifiedAddressing ? "Yes" : "No") << " / "
                  << (p.ECCEnabled ? "Yes" : "No") << " / "
                  << (p.concurrentKernels ? "Yes" : "No") << "\n";

        // Theoretical max resident threads (not a launch limit, but a concurrency ceiling):
        const int maxThreadsResident = p.maxThreadsPerMultiProcessor * p.multiProcessorCount;
        std::cerr << "  Theoretical resident threads (all SMs): " << maxThreadsResident << "\n";

        // Demonstrate occupancy for a chosen block size on the dummy kernel
        int activeBlocksPerSM = 0;
        st = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocksPerSM, dummy_kernel, blockSize, /*dynamicSMem*/ 0);
        if (st == cudaSuccess) {
            int warpsPerBlock = (blockSize + p.warpSize - 1) / p.warpSize;
            int maxWarpsPerSM  = p.maxThreadsPerMultiProcessor / p.warpSize;
            double occ = 100.0 * (activeBlocksPerSM * blockSize) / p.maxThreadsPerMultiProcessor;
            std::cerr << "  Occupancy demo (block " << blockSize << "):\n"
                      << "    Max active blocks / SM : " << activeBlocksPerSM << "\n"
                      << "    Warps/block, warps/SM  : " << warpsPerBlock << ", " << maxWarpsPerSM << "\n"
                      << "    Approx occupancy       : " << std::fixed << std::setprecision(1) << occ << "%\n";
        } else {
            std::cerr << "  Occupancy demo unavailable: " << cudaGetErrorString(st) << "\n";
        }

        // If user provided a problem size, show how to pick grid dims.
        if (problemSize > 0) {
            long long threadsPerBlock = blockSize;
            long long blocksNeeded = (problemSize + threadsPerBlock - 1) / threadsPerBlock;

            // Respect device's grid dimension limit in X (simplest 1D example)
            long long maxGridX = p.maxGridSize[0];
            if (blocksNeeded > maxGridX) {
                // Show a 2D grid decomposition as an example
                long long gridX = std::min<long long>(maxGridX, static_cast<long long>(std::ceil(std::sqrt((double)blocksNeeded))));
                long long gridY = (blocksNeeded + gridX - 1) / gridX;
                gridY = std::min<long long>(gridY, p.maxGridSize[1]);

                std::cerr << "  Launch planning for N=" << problemSize << " with block=" << blockSize << ":\n"
                          << "    Blocks needed          : " << blocksNeeded << "\n"
                          << "    1D grid exceeds limit; example 2D grid:\n"
                          << "      dim3 grid(" << gridX << ", " << gridY << ", 1), dim3 block(" << blockSize << ", 1, 1)\n";
                long long totalThreads = gridX * gridY * threadsPerBlock;
                std::cerr << "    Total launched threads : " << totalThreads << " (>= N)\n";
            } else {
                std::cerr << "  Launch planning for N=" << problemSize << " with block=" << blockSize << ":\n"
                          << "    Blocks needed          : " << blocksNeeded << "\n"
                          << "    Suggested launch       : dim3 grid(" << blocksNeeded << ",1,1), "
                          << "dim3 block(" << blockSize << ",1,1)\n"
                          << "    Total launched threads : " << (blocksNeeded * threadsPerBlock) << " (>= N)\n";
            }

            // Show a simple heuristic for "enough blocks": e.g., 4x–8x SMs to help latency hiding
            long long minUsefulBlocks = std::max( (long long)smCount * 4, (long long)smCount ); // heuristic
            std::cerr << "    Heuristic (latency hiding): aim for >= " << minUsefulBlocks
                      << " total blocks across the grid when feasible.\n";
        }

        std::cerr << std::string(72, '-') << "\n";
    }

    return 0;
}

