# micrograd-cpp

a micrograd implementation in c++

[micrograd](https://github.com/karpathy/micrograd) is an autograd engine that implements backpropagation (reverse-mode autodiff) over a dynamically built DAG

## setup

clone the repo
```bash
git clone --recursive https://github.com/puravparab/micrograd-cpp.git
cd micrograd-cpp
```

## usage

run with bash script
```bash
chmod +x run.sh
./run.sh
```

or
```bash
g++ -std=c++14 -o build/micrograd main.cpp engine.cpp nn.cpp
```

## testing

build google test
```bash
chmod +x build_gtest.sh
./build_gtest.sh
```

compile and run tests
```bash
chmod +x run_tests.sh
./run_tests.sh
```