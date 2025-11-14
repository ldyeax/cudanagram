PQXX_PREFIX := $(CURDIR)/external/build
PQXX_INC    := $(PQXX_PREFIX)/include
PQXX_LIB    := $(PQXX_PREFIX)/lib

GPP = g++
GPP_FLAGS = -O3 -march=native -mavx2 -mfma -std=c++17 -I$(PQXX_INC) -Wno-write-strings -Wno-deprecated-declarations
GPP_LDFLAGS = -L$(PQXX_LIB) -Wl,-rpath,$(PQXX_LIB) -lpqxx -lpq -lncurses -ltinfo -lsqlite3

TEST_DICTIONARY = dictionary_test
TEST_DICTIONARY_O = dictionary_test.o
TEST_DICTIONARY_SRC = dictionary_test.cpp
TEST_AVX = test_avx
TEST_AVX_SRC = test_avx.cpp

AVX_SRC = avx.cpp
AVX_O = avx.o
DB_SRC = database.cpp
DB_O = database.o

LSQLITE3 = -lsqlite3

NVCC = nvcc
NVCC_CFLAGS = -ccbin g++ --expt-relaxed-constexpr -arch=sm_86 -std=c++17 -I$(PQXX_INC) -Xcompiler -Wno-write-strings -Xcompiler -O3 -O3 -use_fast_math
LDFLAGS = -ltinfo -L$(PQXX_LIB) -lpq -lpqxx -lsqlite3

TARGET = cudanagram
SRC = cudanagram.cu
TEST_CAP = "capabilities_test"
TEST_CAP_SRC = capabilities_test.cu
TEST_DB_SRC = database_test.cpp
TEST_DB = database_test

DICTIONARY_CU = dictionary.cu
DICTIONARY_CPP = dictionary.cpp

NVCC_XLINKER = -Xlinker -rpath -Xlinker $(PQXX_LIB)

WORKER_CPU_O = worker_cpu.o
WORKER_CPU_SRC = worker_cpu.cpp
WORKER_GPU_O = worker_gpu.o
WORKER_GPU_SRC = worker_gpu.cu
FM_CU = frequency_map.cu
FM_CPP = frequency_map.cpp
FM_O = frequency_map.o
DICTIONARY_O = dictionary.o

ifdef TEST_WORKER_GPU
    GPP_FLAGS  += -DTEST_WORKER_GPU
    NVCC_CFLAGS += -DTEST_WORKER_GPU
endif

ifdef TEST_ANAGRAMMER
	GPP_FLAGS  += -DTEST_ANAGRAMMER
	NVCC_CFLAGS += -DTEST_ANAGRAMMER
endif

ifdef CUDANAGRAM_PSQL
	GPP_FLAGS  += -DCUDANAGRAM_PSQL
	NVCC_CFLAGS += -DCUDANAGRAM_PSQL
endif
ifdef CUDANAGRAM_SQLITE
	GPP_FLAGS  += -DCUDANAGRAM_SQLITE
	NVCC_CFLAGS += -DCUDANAGRAM_SQLITE
endif

ifdef CUDANAGRAM_THREADS_PER_CPU_WORKER
	GPP_FLAGS  += -DCUDANAGRAM_THREADS_PER_CPU_WORKER=$(CUDANAGRAM_THREADS_PER_CPU_WORKER)
	NVCC_CFLAGS += -DCUDANAGRAM_THREADS_PER_CPU_WORKER=$(CUDANAGRAM_THREADS_PER_CPU_WORKER)
endif

ifdef SQLITE_TEST
	GPP_FLAGS  += -DSQLITE_TEST
	NVCC_CFLAGS += -DSQLITE_TEST
endif


all: $(TARGET)

$(TARGET): $(SRC)
	$(GPP) $(GPP_FLAGS) -c $(AVX_SRC)
	$(GPP) $(GPP_FLAGS) -c $(DB_SRC)
	$(GPP) $(GPP_FLAGS) -c $(WORKER_CPU_SRC)
	$(NVCC) $(NVCC_CFLAGS) -rdc=true -c $(DICTIONARY_CU) $(WORKER_GPU_SRC)
	$(NVCC) $(NVCC_CFLAGS) -rdc=true -c $(FM_CU)
	$(NVCC) $(NVCC_CFLAGS) \
		$(AVX_O) $(DB_O) $(WORKER_CPU_O) $(WORKER_GPU_O) \
		$(DICTIONARY_O) $(FM_O) \
		-o $(TARGET) \
		$(SRC) anagrammer.cpp worker.cpp \
		$(LDFLAGS)

clean:
	rm -f $(TARGET)
	rm -f capabilities_test
	rm -f database_test
	rm -f test_avx
	rm -f dictionary_test
	rm -f worker_cpu_test
	rm -f sqlite_test
	rm -f *.o

pqxx:
	bash ./build_pqxx.sh

avx_test:
	$(GPP) -c $(AVX_SRC) $(GPP_LDFLAGS)
	$(GPP) $(GPP_FLAGS) -o $(TEST_AVX) $(TEST_AVX_SRC) $(GPP_LDFLAGS)

capabilities_test:
	$(NVCC) $(NVCC_CFLAGS) -o capabilities_test capabilities_test.cu $(LDFLAGS)

database_test:
	$(GPP) $(GPP_FLAGS) -c $(AVX_SRC)
	#$(GPP) $(GPP_FLAGS) -c $(DB_SRC) -Wl,-rpath,$(PQXX_LIB)
	$(GPP) $(GPP_FLAGS) -c $(DB_SRC) $(GPP_LDFLAGS)
	$(NVCC) $(NVCC_CFLAGS) $(DB_O) $(AVX_O) -o $(TEST_DB) dictionary.cu frequency_map.cu database_test.cpp $(NVCC_XLINKER) $(LDFLAGS) -DTEST_DB

dictionary_test:
	$(GPP) $(GPP_FLAGS) \
		-c $(AVX_SRC) \
		$(GPP_LDFLAGS)
	$(GPP) $(GPP_FLAGS) \
		$(AVX_O) \
		-o $(TEST_DICTIONARY) \
		dictionary_test.cpp dictionary.cpp frequency_map.cpp \
		$(GPP_LDFLAGS) \
		-DDICTIONARY_DEBUG

worker_cpu_test:
	$(GPP) $(GPP_FLAGS) -c $(AVX_SRC)
	$(GPP) $(GPP_FLAGS) -c $(DB_SRC)
	$(GPP) $(GPP_FLAGS) -c worker_cpu.cpp
	$(GPP) $(GPP_FLAGS) -c worker_gpu.cpp
	$(GPP) $(GPP_FLAGS) \
		$(AVX_O) $(DB_O) $(WORKER_CPU_O) $(WORKER_GPU_O) \
		-o worker_cpu_test \
		anagrammer.cpp worker.cpp worker_cpu_test.cpp dictionary.cpp frequency_map.cpp  \
		$(GPP_LDFLAGS)

sqlite_test:
	$(GPP) $(GPP_FLAGS) \
		sqlite_test.cpp \
		$(DB_SRC) \
		$(DICTIONARY_CPP) \
		$(FM_CPP) \
		$(AVX_SRC) \
		-o sqlite_test \
		$(GPP_FLAGS) \
		$(LSQLITE3)
