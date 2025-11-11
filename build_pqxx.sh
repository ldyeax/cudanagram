git submodule update --init --recursive
(rm -rf ./external/build || true) 2>/dev/null
cd external/libpqxx
make clean
CXXFLAGS='-O3' ./configure --prefix=$(pwd)/../build
make -j$(nproc)
make install
