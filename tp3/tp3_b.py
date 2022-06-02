from __future__ import division

import math
import random

import matplotlib.pyplot as plt
import numpy as np
from linear_algebra import dot


def saida_adaline(pesos, entradas):
    y = dot(pesos, entradas)
    return y


def linear(sinapses):
    pesos_sinapses = sinapses
    taxa_aprendizagem = 0.09
    termo_proporcionalidade = 1
    seno = [i for i in range(45)]
    coseno = [i for i in range(45)]
    coeficiente = [i for i in range(45)]
    entradas = [i for i in range(45)]
    saida_parcial = [i for i in range(45)]

    for x in range(45):
        f[x] = -math.pi + 0.555 * math.sin(math.pi/180 * x) + 2.657 * math.cos(
            math.pi/180 * x) + 0.674 * math.pi/180 * x
        seno[x] = math.sin(math.pi/180*x)
        coseno[x] = math.cos(math.pi/180*x)
        coeficiente[x] = math.pi/180*x
        entradas[x] = [termo_proporcionalidade,
                       seno[x], coseno[x], coeficiente[x]]
        saida_parcial[x] = saida_adaline(pesos_sinapses, entradas[x])
        pesos_sinapses[0] = pesos_sinapses[0] + taxa_aprendizagem * \
            ((f[x] - saida_parcial[x])*(f[x] - saida_parcial[x])) * \
            0.5 * termo_proporcionalidade
        pesos_sinapses[1] = pesos_sinapses[1] + taxa_aprendizagem * \
            ((f[x] - saida_parcial[x])*(f[x] - saida_parcial[x])) * \
            0.5 * math.sin(math.pi/180 * x)
        pesos_sinapses[2] = pesos_sinapses[2] + taxa_aprendizagem * \
            ((f[x] - saida_parcial[x])*(f[x] - saida_parcial[x])) * \
            0.5 * math.cos(math.pi/180 * x)
        pesos_sinapses[3] = pesos_sinapses[3] + taxa_aprendizagem * \
            ((f[x] - saida_parcial[x])*(f[x] - saida_parcial[x])) * \
            0.5 * math.pi/180 * x

    return pesos_sinapses, saida_parcial


def teste_generalizacao(sinapses):
    pesos_sinapses = sinapses
    termo_proporcionalidade = 1
    seno = [i for i in range(359)]
    coseno = [i for i in range(359)]
    coeficiente = [i for i in range(359)]
    entradas = [i for i in range(359)]
    saida_parcial = [i for i in range(359)]

    for x in range(359):
        f[x] = -math.pi + 0.565 * (math.sin(math.pi/180)) + 2.657 * (
            math.cos(math.pi/180)) + 0.674 * (math.pi/180) * random.random()
        seno[x] = (math.sin(math.pi/180))
        coseno[x] = (math.cos(math.pi/180))
        coeficiente[x] = (math.pi/180) * random.random()
        entradas[x] = [termo_proporcionalidade,
                       seno[x], coseno[x], coeficiente[x]]
        saida_parcial[x] = saida_adaline(pesos_sinapses, entradas[x])

    saida = saida_parcial
    return sinapses, saida


t = np.arange(0, 359, 1)
seno_0 = 0 + np.sin(np.pi/180 * t)
coseno_0 = 0 + np.cos(np.pi/180 * t)
coeficiente_0 = 0 + np.pi/180 * t

f = -np.pi + 0.565 * seno_0 + 2.657 * coseno_0 + 0.674 * coeficiente_0
neuronio = [-2.8013, 0.393, 1.902, 0.429]

for _ in range(30):
    neuronio, funcao_saida = linear(neuronio)
    print(neuronio)

fig, ax = plt. subplots()
ax.plot(t, seno_0)
ax.plot(t, coseno_0)
ax.plot(t, coeficiente_0)
ax.plot(t, f)
ax.set(xlabel='Angulo', ylabel='Funcoes', title='Adaline B')
ax.grid()
fig.savefig("./tp3/img/adaline_B.png")
neuronio, funcao = teste_generalizacao(neuronio)
ax.plot(t, funcao)
plt.show()
fig.savefig("./tp3/img/adaline_B_1.png")
