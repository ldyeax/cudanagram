PQXX_PREFIX := $(CURDIR)/external/build
PQXX_INC    := $(PQXX_PREFIX)/include
PQXX_LIB    := $(PQXX_PREFIX)/lib

GPP = g++
GPP_FLAGS = -O3 -march=native -mavx2 -mfma -std=c++17 -I$(PQXX_INC) -Wno-write-strings -Wno-deprecated-declarations
GPP_LDFLAGS = -L$(PQXX_LIB) -Wl,-rpath,$(PQXX_LIB) -lpqxx -lpq -lncurses -ltinfo 

TEST_DICTIONARY = dictionary_test
TEST_DICTIONARY_O = dictionary_test.o
TEST_DICTIONARY_SRC = dictionary_test.cpp
TEST_AVX = test_avx
TEST_AVX_SRC = test_avx.cpp

AVX_SRC = avx.cpp
AVX_O = avx.o
DB_SRC = database.cpp
DB_O = database.o

NVCC = nvcc
NVCC_CFLAGS = -ccbin g++ --expt-relaxed-constexpr -arch=sm_86 -std=c++17 -I$(PQXX_INC)
LDFLAGS = -ltinfo -L$(PQXX_LIB) -lpq -lpqxx

TARGET = cudanagram
SRC = main.cu
TEST_CAP = "capabilities_test"
TEST_CAP_SRC = capabilities_test.cu
TEST_DB_SRC = database_test.cpp
TEST_DB = database_test

DICTIONARY_CU = dictionary.cu

NVCC_XLINKER = -Xlinker -rpath -Xlinker $(PQXX_LIB)

WORKER_CPU_O = worker_cpu.o
FM_CU = frequency_map.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(GPP) $(GPP_FLAGS) -o $(AVX_O) $(AVX_SRC) $(GPP_LDFLAGS)
	$(GPP) $(GPP_FLAGS) -o $(DB_O) $(DB_SRC) $(GPP_LDFLAGS)
	$(NVCC) $(NVCC_CFLAGS)  -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
	rm -f capabilities_test
	rm -f database_test
	rm -f test_avx
	rm -f dictionary_test
	rm -f worker_cpu_test
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
	$(GPP) $(GPP_FLAGS) $(AVX_O) $(DB_O) $(WORKER_CPU_O) \
		-o worker_cpu_test \
		worker.cpp worker_cpu_test.cpp dictionary.cpp frequency_map.cpp  \
		$(GPP_LDFLAGS)
