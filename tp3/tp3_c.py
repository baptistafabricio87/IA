from __future__ import division

import math
import random
from linear_algebra import dot

import matplotlib.pyplot as plt
import numpy as np


def saida_adaline(pesos, entradas):
    y = dot(pesos, entradas)
    return y


def linear(sinapses):
    pesos_sinapses = sinapses
    taxa_aprendizagem = 0.1
    trm_proporcionalidade = 1
    seno = [i for i in range(45)]
    coseno = [i for i in range(45)]
    coeficiente = [i for i in range(45)]
    entradas = [i for i in range(45)]
    saida_parcial = [i for i in range(45)]

    for i in range(45):
        f[i] = -math.pi + 0.565 * math.sin(math.pi / 180 * i) + 2.657 * math.cos(
            math.pi / 180 * i) + 0.674 * math.pi / 180 * i
        seno[i] = math.sin(math.pi / 180 * i)
        coseno[i] = math.cos(math.pi / 180 * i)
        coeficiente[i] = math.pi / 180 * i
        entradas[i] = [trm_proporcionalidade, seno[i], coseno[i], coeficiente[i]]
        saida_parcial[i] = saida_adaline(pesos_sinapses, entradas[i])
        pesos_sinapses[0] = pesos_sinapses[0] + taxa_aprendizagem * (
                (f[i] - saida_parcial[i]) * (f[i] - saida_parcial[i])) * 0.5 * trm_proporcionalidade
        pesos_sinapses[1] = pesos_sinapses[1] + taxa_aprendizagem * (
                (f[i] - saida_parcial[i]) * (f[i] - saida_parcial[i])) * 0.5 * math.sin(math.pi / 180 * i)
        pesos_sinapses[2] = pesos_sinapses[2] + taxa_aprendizagem * (
                (f[i] - saida_parcial[i]) * (f[i] - saida_parcial[i])) * 0.5 * math.cos(math.pi / 180 * i)
        pesos_sinapses[3] = pesos_sinapses[3] + taxa_aprendizagem * (
                (f[i] - saida_parcial[i]) * (f[i] - saida_parcial[i])) * 0.5 * math.pi / 180 * i

    return pesos_sinapses, saida_parcial


def teste_generalizacao(sinapses):
    pesos_sinapses = sinapses

    trm_proporcionalidade = 1
    seno = [i for i in range(359)]
    coseno = [i for i in range(359)]
    coeficiente = [i for i in range(359)]
    entradas = [i for i in range(359)]
    saida_parcial = [i for i in range(359)]

    for i in range(359):
        f[i] = -math.pi + 0.565 * (math.sin(math.pi / 180)) + 2.657 * (math.cos(math.pi / 180)) + 0.674 * (
                math.pi / 180) * random.random()
        seno[i] = (math.sin(math.pi / 180))
        coseno[i] = (math.cos(math.pi / 180))
        coeficiente[i] = (math.pi / 180) * random.random() * 2.4
        entradas[i] = [trm_proporcionalidade, seno[i], coseno[i], coeficiente[i]]
        saida_parcial[i] = saida_adaline(pesos_sinapses, entradas[i])

    saida = saida_parcial
    return sinapses, saida


t = np.arange(0, 359, 1)
seno_0 = 0 + np.sin(np.pi / 180 * t)
coseno_0 = 0 + np.cos(np.pi / 180 * t)
coeficiente_0 = 0 + np.pi / 180 * t
trm_proporcionalidade = 1
f = -np.pi + 0.565 * seno_0 + 2.657 * coseno_0 + 0.674 * coeficiente_0
neuronio = [-2.4013, 0.393, 1.902, 0.429]

for _ in range(20):
    neuronio, funcao_saida = linear(neuronio)
    print(neuronio)

fig, ax = plt.subplots()
ax.plot(t, seno_0)
ax.plot(t, coseno_0)
ax.plot(t, coeficiente_0)
ax.plot(t, f)
ax.set(xlabel='Angulo', ylabel='Funcao', title='Adaline C')
ax.grid()
fig.savefig("adaline_C.png")
neuronio, funcao = teste_generalizacao(neuronio)
ax.plot(t, funcao)
plt.show()
