# g++ -O3 -march=native -mavx2 -mfma -Wall -Wextra -std=c++20 test_avx.cpp -o test_avx && ./test_avx

GPP = g++
GPP_FLAGS = -O3 -march=native -mavx2 -mfma -std=c++20
TEST_AVX = test_avx
TEST_AVX_SRC = test_avx.cpp
AVX_SRC = avx.cpp
AVX_O = avx.o
DB_SRC = database.cpp
DB_O = database.o
AVX_2 = $(GPP_FLAGS) -c $(AVX_SRC)
DB_2 = $(GPP_FLAGS) -c $(DB_SRC)
CC = nvcc
CFLAGS = --expt-relaxed-constexpr -Wno-deprecated-gpu-targets -arch=native -std=c++20
LDFLAGS = -lncurses -ltinfo -lpqxx -lpq
TARGET = cudanagram
SRC = main.cu
TEST_CAP = "capabilities_test"
TEST_CAP_SRC = capabilities_test.cu
TEST_DB_SRC = database_test.cpp
TEST_DB = database_test
CUFILES := $(wildcard *.cu)
CUFILES := $(filter-out $(SRC), $(CUFILES))
CUFILES := $(filter-out $(TEST_CAP_SRC), $(CUFILES))
CPPFILES := $(wildcard *.cpp)
CPPFILES := $(filter-out $(TEST_DB_SRC), $(CPPFILES))
CPPFILES := $(filter-out $(AVX_SRC), $(CPPFILES))
CPPFILES := $(filter-out $(TEST_AVX_SRC), $(CPPFILES))

all: $(TARGET)

$(TARGET): $(SRC)
	$(GPP) $(GPP_FLAGS) -o $(AVX_O) $(AVX_SRC) $(LDFLAGS)
	$(GPP) $(GPP_FLAGS) -o $(DB_O) $(DB_SRC) $(LDFLAGS)
	$(CC) $(CFLAGS)  -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
	rm -f capabilities_test
	rm -f database_test
	rm -f $(AVX_O)

avx_test:
	$(GPP) $(AVX_2) $(LDFLAGS)
	$(GPP) $(GPP_FLAGS) -o $(TEST_AVX) $(TEST_AVX_SRC) $(LDFLAGS)

capabilities_test:
	$(CC) $(CFLAGS) -o capabilities_test capabilities_test.cu $(LDFLAGS)

database_test:
	$(GPP) $(AVX_2) $(LDFLAGS)
	$(GPP) $(DB_2) $(LDFLAGS)
	$(CC) $(CFLAGS) $(DB_O) $(AVX_O) -o $(TEST_DB) $(CUFILES) $(TEST_DB_SRC) $(LDFLAGS)
