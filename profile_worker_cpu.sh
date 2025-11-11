make clean
CXX="g++"
CXXFLAGS="-O3 -march=native -mavx2 -mfma -std=c++17 -I/root/cudanagram/external/build/include \
          -Wno-write-strings -Wno-deprecated-declarations -g -fno-omit-frame-pointer"
LDFLAGS="-L/root/cudanagram/external/build/lib -Wl,-rpath,/root/cudanagram/external/build/lib \
         -lpqxx -lpq -lncurses -ltinfo"

$CXX $CXXFLAGS -c avx.cpp
$CXX $CXXFLAGS -c database.cpp
$CXX $CXXFLAGS -c worker_cpu.cpp
$CXX $CXXFLAGS avx.o database.o worker_cpu.o \
      -o worker_cpu_test \
      worker.cpp worker_cpu_test.cpp dictionary.cpp frequency_map.cpp \
      $LDFLAGS

perf record --call-graph=dwarf -F 999 -- ./worker_cpu_test
#perf report
