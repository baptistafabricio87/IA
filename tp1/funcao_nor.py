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

print("PERCEPTRON IMPLEMENTANDO FUNcaO BOOLEANA NOR")
print("0 x 0 =", saida0)
print("0 x 1 =", saida1)
print("1 x 0 =", saida2)
print("1 x 1 =", saida3)
