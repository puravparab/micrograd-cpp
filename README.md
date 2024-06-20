# micrograd-cpp

a micrograd implementation in c++

[micrograd](https://github.com/karpathy/micrograd) is an autograd engine that implements backpropagation (reverse-mode autodiff) over a dynamically built DAG

## run

```bash
g++ -o micrograd main.cpp engine.cpp nn.cpp -std=c++11
```