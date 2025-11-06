CC = nvcc
CFLAGS = --expt-relaxed-constexpr -Wno-deprecated-gpu-targets -arch=native
LDFLAGS = -lncurses -ltinfo
TARGET = cudanagram
SRC = main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
