# Trabalho 1 de Inteligência Artificial – Manhã

---
| NOME                               | MATRICULA     |
| ---------------------------------- | ------------- |
| Fabricio Baptista de Castro        | 0050481821007 |
| Mario Celso Zanin                  | 0050481921023 |
---
---

## Código AND

```python
from linear_algebra import dot


def step_function(x):
    return 1 if x >= 1 else 0


def perceptron_output(weights, bias, x):
    calculation = dot(weights, x) + bias
    return step_function(calculation)


x0 = [0, 0]
x1 = [0, 1]
x2 = [1, 0]
x3 = [1, 1]

weights = [1, 1]
bias = -1

saida0 = perceptron_output(weights, bias, x0)
saida1 = perceptron_output(weights, bias, x1)
saida2 = perceptron_output(weights, bias, x2)
saida3 = perceptron_output(weights, bias, x3)

print("PERCEPTRON IMPLEMENTANDO FUNCAO BOOLEANA AND")
print("0 x 0 =", saida0)
print("0 x 1 =", saida1)
print("1 x 0 =", saida2)
print("1 x 1 =", saida3)

```

---

## Console AND

```powershell

PS D:\workspace\IA> & C:/Users/bapti/AppData/Local/Programs/Python/Python310/python.exe d:/workspace/IA/tp1/funcao_and.py
PERCEPTRON IMPLEMENTANDO FUNCAO BOOLEANA AND
0 x 0 = 0
0 x 1 = 0
1 x 0 = 0
1 x 1 = 1
```

---
---

## Código OR

```python

from linear_algebra import dot


def step_function(x):
    return 1 if x >= 1 else 0


def perceptron_output(weights, bias, x):
    calculation = dot(weights, x) + bias
    return step_function(calculation)


x0 = [0, 0]
x1 = [0, 1]
x2 = [1, 0]
x3 = [1, 1]

weights = [2, 2]
bias = 0

saida0 = perceptron_output(weights, bias, x0)
saida1 = perceptron_output(weights, bias, x1)
saida2 = perceptron_output(weights, bias, x2)
saida3 = perceptron_output(weights, bias, x3)

print("PERCEPTRON IMPLEMENTANDO FUNCAO BOOLEANA OR")
print("0 x 0 =", saida0)
print("0 x 1 =", saida1)
print("1 x 0 =", saida2)
print("1 x 1 =", saida3)

```

---

## Console OR

```powershell

PS D:\workspace\IA> & C:/Users/bapti/AppData/Local/Programs/Python/Python310/python.exe d:/workspace/IA/tp1/funcao_or.py
PERCEPTRON IMPLEMENTANDO FUNCAO BOOLEANA OR
0 x 0 = 0
0 x 1 = 1
1 x 0 = 1
1 x 1 = 1
```

---
---

## Código NAND

```python

from linear_algebra import dot


def step_function(x):
    return 1 if x >= 1 else 0


def perceptron_output(weights, bias, x):
    calculation = dot(weights, x) + bias
    return step_function(calculation)


x0 = [0, 0]
x1 = [0, 1]
x2 = [1, 0]
x3 = [1, 1]

weights = [-2, -2]
bias = 3

saida0 = perceptron_output(weights, bias, x0)
saida1 = perceptron_output(weights, bias, x1)
saida2 = perceptron_output(weights, bias, x2)
saida3 = perceptron_output(weights, bias, x3)

print("PERCEPTRON IMPLEMENTANDO FUNCAO BOOLEANA NAND")
print("0 x 0 =", saida0)
print("0 x 1 =", saida1)
print("1 x 0 =", saida2)
print("1 x 1 =", saida3)

```

---

## Console NAND

```powershell

PS D:\workspace\IA> & C:/Users/bapti/AppData/Local/Programs/Python/Python310/python.exe d:/workspace/IA/tp1/funcao_nand.py
PERCEPTRON IMPLEMENTANDO FUNCAO BOOLEANA NAND0 x 0 = 1
0 x 1 = 1
1 x 0 = 1
1 x 1 = 0
```

---
---

## Código NOR

```python

from linear_algebra import dot


def step_function(x):
    return 1 if x >= 1 else 0


def perceptron_output(weights, bias, x):
    calculation = dot(weights, x) + bias
    return step_function(calculation)


x0 = [0, 0]
x1 = [0, 1]
x2 = [1, 0]
x3 = [1, 1]

weights = [-2, -2]
bias = 1

saida0 = perceptron_output(weights, bias, x0)
saida1 = perceptron_output(weights, bias, x1)
saida2 = perceptron_output(weights, bias, x2)
saida3 = perceptron_output(weights, bias, x3)

print("PERCEPTRON IMPLEMENTANDO FUNCAO BOOLEANA NOR")
print("0 x 0 =", saida0)
print("0 x 1 =", saida1)
print("1 x 0 =", saida2)
print("1 x 1 =", saida3)

```

---

## Console NOR

```powershell

PS D:\workspace\IA> & C:/Users/bapti/AppData/Local/Programs/Python/Python310/python.exe d:/workspace/IA/tp1/funcao_nor.py
PERCEPTRON IMPLEMENTANDO FUNCAO BOOLEANA NOR
0 x 0 = 1
0 x 1 = 0
1 x 0 = 0
1 x 1 = 0

```

---
---
