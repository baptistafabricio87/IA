# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 23:01:30 2022

@author: MarioCelso
"""
from __future__ import division

from linear_algebra import dot

import matplotlib.pyplot as plt
import numpy as np
import math #pg 56 - Summerfield
import math, random

def sigmoid(t):
    return ((2 / (1 + math.exp(-t)))-1)

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    outputs = []
    for layer in neural_network:
        input_with_bias = input_vector + [1]             # adiciona bias à entrada
        output = [neuron_output(neuron, input_with_bias) # calcula a saída do neurônio 
                  for neuron in layer]                   # para cada camada
        #print(output)
        outputs.append(output)
        
        # a saída de uma camada de neurônio é a entrada da próxima camada
        input_vector = output
   
    return outputs

alpha = 0.08

def backpropagate(network, input_vector, target):
    # feed_forward calcula a saída dos neurônios usando sigmóide
    hidden_outputs, outputs = feed_forward(network, input_vector)
    # 0.5 *alpha* (1 + output) * (1 - output) cálculo de derivada de sigmóide
    output_deltas = [0.5 * (1 + output) * (1 - output) * (output - target[i]) * alpha
                     for i, output in enumerate(outputs)]
    
    # ajuste dos pesos sinápticos para camadas de saída (network[-1])
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output
    
    # 0.5 *alpha* (1 +output)*(1-output) cálculo de derivada da sigmóide
    # retro-programação do erro para camadas intermediárias
    hidden_deltas = [ 0.5 * alpha * (1 + hidden_output) * (1 - hidden_output) *
                      dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]
    
    # ajuste dos pesos sinápticos para camadas intermediárias (network[0])
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input
 
def seno(x): # função a ser aproximada pela rede neural
       seno = [(math.sin(2*math.pi/180*x)*math.sin(math.pi/180*x))]
       # seno é uma lista
       # [(0.8+(math.sin(math.pi/180*x)*math.sin(2*math.pi/180*x)))*0.5]
       return [seno]
   
def predict(inputs):
     return feed_forward(network, inputs)[-1]

inputs = []
targets = []
for x in range(360):
    seno_a = seno(x)
    
# TREINAMENTO DA REDE NEURAL
random.seed(0) # valores iniciais de pesos sinápticos
input_size = 1 # dimensão do vetor de entrada
num_hidden = 6 # número de neurônios na camada intermediária
output_size = 1 # dimensão das camadas de saída = 1 neurônio

"""
inserindo manualmente os vetores relativos à camada intermediária e a saída da Rede Neural
hidden_layer = [[-0.085, -0.09], [-0.033, -0.08], [-0.074, -0.063], [-0.075, -0.065], [-0.088, -0.076], [-0.077, -0.072]]
output_layer = [[0.082, -0.09, 0.064, -0.08, 0.084, -0.075, 0.099]] """

# cada neurônio da camada intermediária tem um peso sináptico associado à entrada
# e adicionado o peso do bias
hidden_layer = [[random.random() for __ in range(input_size + 1)]
                for __ in range(num_hidden)]

#print(hidden_layer)
# neurônio de saída tem um peso sináptico associado a cada neurôio da camada intermediária
# e adicionado o peso do bias
output_layer = [[random.random() for __ in range(num_hidden +1)]
                for __ in range(output_size)]

# a rede inicializa com pesos sinápticos randômicos
network = [hidden_layer, output_layer]
# print(network)
for __ in range(300): # número de ciclos de treinamento
    for x in range(360):
        inputs = seno(x)
        targets = seno(x)
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)

#TERINAMENTO DA REDE NEURAL
# formação do gráfico
fig, ax = plt.subplots()
ax.set(xlabel='Angulo (º)', ylabel='Função sen(x)*sen(2x)',
       title='APROXIMAÇÃO FUNCIONAL (Ciclos:300|Neuronio:6|Alpha:0.08)')
ax.grid()
t = np.arange(0, 360,1) 

#teste da rede através de predict()
saida = []
for x in range(360):
    inputs = seno(x)
    targets = seno(x)
    for input_vector, target_vector in zip(inputs, targets):
        sinal_saida = predict(input_vector)
        saida.extend(sinal_saida)
        
entrada = []
for x in range(360):
    entrada += seno(x) # criando o arranjo da função de entrada para o gráfico
ax.plot(t, entrada)
ax.plot(t, saida)

print (f'Comando Entrada {hidden_layer}')
print (f'Comando Saída {output_layer}')
   
plt.show()
fig.savefig('aprox_func_base')
