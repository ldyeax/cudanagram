PQXX_PREFIX := $(CURDIR)/external/build
PQXX_INC    := $(PQXX_PREFIX)/include
PQXX_LIB    := $(PQXX_PREFIX)/lib

GPP = g++
GPP_FLAGS = -O3 -march=native -mavx2 -mfma -std=c++17 -I$(PQXX_INC)
GPP_LDFLAGS = -lpq -lncurses -ltinfo -L$(PQXX_LIB) -Wl,-rpath,$(PQXX_LIB) 

TEST_DICTIONARY = dictionary_test
TEST_DICTIONARY_O = dictionary_test.o
TEST_DICTIONARY_SRC = dictionary_test.cpp
TEST_AVX = test_avx
TEST_AVX_SRC = test_avx.cpp
AVX_SRC = avx.cpp
AVX_O = avx.o
DB_SRC = database.cpp
DB_O = database.o
AVX_2 = $(GPP_FLAGS) -c $(AVX_SRC)
DB_2 = $(GPP_FLAGS) -c $(DB_SRC)

CC = nvcc
CFLAGS = -ccbin g++ --expt-relaxed-constexpr -arch=sm_86 -std=c++17 -I$(PQXX_INC)
LDFLAGS = -ltinfo -L$(PQXX_LIB) -lpq -lpqxx



TARGET = cudanagram
SRC = main.cu
TEST_CAP = "capabilities_test"
TEST_CAP_SRC = capabilities_test.cu
TEST_DB_SRC = database_test.cpp
TEST_DB = database_test

all: $(TARGET)

$(TARGET): $(SRC)
	$(GPP) $(GPP_FLAGS) -o $(AVX_O) $(AVX_SRC) $(GPP_LDFLAGS)
	$(GPP) $(GPP_FLAGS) -o $(DB_O) $(DB_SRC) $(GPP_LDFLAGS)
	$(CC) $(CFLAGS)  -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
	rm -f capabilities_test
	rm -f database_test
	rm -f $(AVX_O)
	rm -f test_avx
	rm -f dictionary_test

pqxx:
	bash ./build_pqxx.sh

avx_test:
	$(GPP) $(AVX_2) $(GPP_LDFLAGS)
	$(GPP) $(GPP_FLAGS) -o $(TEST_AVX) $(TEST_AVX_SRC) $(GPP_LDFLAGS)

capabilities_test:
	$(CC) $(CFLAGS) -o capabilities_test capabilities_test.cu $(LDFLAGS)

database_test:
	$(GPP) $(GPP_FLAGS) -c $(AVX_SRC)
	$(GPP) $(GPP_FLAGS) -c $(DB_SRC) -Wl,-rpath,$(PQXX_LIB) 

	$(CC) $(CFLAGS) $(DB_O) $(AVX_O) -o $(TEST_DB) dictionary.cu frequency_map.cu database_test.cpp -Xlinker -rpath -Xlinker /root/cudanagram/external/build/lib $(LDFLAGS) 

dictionary_test:
	$(GPP) $(AVX_2) $(GPP_LDFLAGS)
	$(GPP) $(GPP_FLAGS) $(AVX_O) -o $(TEST_DICTIONARY) $(TEST_DICTIONARY_SRC) dictionary.cpp frequency_map.cpp $(GPP_LDFLAGS)
