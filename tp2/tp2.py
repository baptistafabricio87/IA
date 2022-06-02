from __future__ import division

import matplotlib.pyplot as plt
from linear_algebra import dot


def degrau(x):
    return 1 if x >= 0 else 0


def saida_perceptron(pesos, entradas):
    y = dot(pesos, entradas)
    return degrau(y)


def ajustes(sinapses, entradas, saida):
    taxa_aprendizagem = 0.0981
    saida_parcial = saida_perceptron(sinapses, entradas)

    for j in range(3):
        sinapses[j] = sinapses[j] + taxa_aprendizagem * \
            (saida[0] - saida_parcial) * entradas[j]
    saida = saida_parcial
    return sinapses, saida


def teste_generalizacao(sinapses, entradas, saida):
    saida_parcial = saida_perceptron(sinapses, entradas)
    saida = saida_parcial
    return sinapses, saida


neuronio = [0.22, -0.33, 0.44]
padrao_0_0 = [-1, 0.2, 0.5]
padrao_0_1 = [-1, 0.1, 0.7]
padrao_0_2 = [-1, 0.3, 0.4]
padrao_1_0 = [-1, 0.9, 0.7]
padrao_1_1 = [-1, 0.7, 0.6]
padrao_1_2 = [-1, 0.9, 0.7]
saida0 = [0]
saida1 = [1]

for x in range(20):
    neuronio, saida_0 = ajustes(neuronio, padrao_0_0, saida0)
    print(neuronio, "Saida 0 = ", saida_0)
    neuronio, saida_0 = ajustes(neuronio, padrao_0_1, saida0)
    print(neuronio, "Saida 0 = ", saida_0)
    neuronio, saida_0 = ajustes(neuronio, padrao_0_2, saida0)
    print(neuronio, "Saida 0 = ", saida_0)
    neuronio, saida_1 = ajustes(neuronio, padrao_1_0, saida1)
    print(neuronio, "Saida 1 = ", saida_1)
    neuronio, saida_1 = ajustes(neuronio, padrao_1_1, saida1)
    print(neuronio, "Saida 1 = ", saida_1)
    neuronio, saida_1 = ajustes(neuronio, padrao_1_2, saida1)
    print(neuronio, "Saida 1 = ", saida_1)

    x = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    y = [-0.125, -0.0125, 0.1, 0.2125, 0.325,
         0.4375, 0.55, 0.6625, 0.775, 0.8875]
    x1 = [0.1, 0.5, 0.3]
    x2 = [0.6, 0.2, 0.3]
    x3 = [0.3, 0.4, 0.6]
    y1 = [0.1, 0.1, 0.3]
    y2 = [0.6, 0.8, 0.9]
    y3 = [0.7, 0.5, 0.8]

    # pontos x e y no gráfico
    plt.scatter(y1, x1)
    plt.scatter(y2, x2)
    plt.scatter(y3, x3)

    # linha que divide
    plt.plot(y, x, color='green', marker='*', linestyle='--')

    plt.title("Indices")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")

    padrao_teste_0 = [-1, 0.2, 0.4]
    padrao_teste_1 = [-1, 0.7, 0.8]
    padrao_teste_2 = [-1, 0.6, 0.3]
    padrao_teste_3 = [-1, 0.1, 0.9]
    padrao_teste_4 = [-1, 0.2, 0.6]
    padrao_teste_5 = [-1, 0.8, 0.1]

    print("TESTE DE GENERALIZACAO")
    neuronio, saida_0 = teste_generalizacao(neuronio, padrao_teste_0, saida0)
    print(neuronio, "entre ano x e y = ", saida_0)
    neuronio, saida_1 = teste_generalizacao(neuronio, padrao_teste_1, saida1)
    print(neuronio, "entre ano w e z = ", saida_1)
    neuronio, saida_0 = teste_generalizacao(neuronio, padrao_teste_2, saida0)
    print(neuronio, "entre ano x e y = ", saida_0)
    neuronio, saida_1 = teste_generalizacao(neuronio, padrao_teste_3, saida1)
    print(neuronio, "entre ano w e z = ", saida_1)
    neuronio, saida_0 = teste_generalizacao(neuronio, padrao_teste_4, saida0)
    print(neuronio, "entre ano x e y = ", saida_0)
    neuronio, saida_1 = teste_generalizacao(neuronio, padrao_teste_5, saida1)
    print(neuronio, "entre ano w e z = ", saida_1)

plt.savefig("./tp2/grafico.png")
plt.show()
