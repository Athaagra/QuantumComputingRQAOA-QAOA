#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 17:39:21 2023

@author: Optimus
"""

import pennylane as qml
from pennylane import numpy as np

np.random.seed(42)

n_wires = 20
graph = [(0,1), (0,2), (1, 3), (3,4), (4,5), (5,6), (6,7), (7,2), (8,1), (9,2), (10,3), (11,4), (12,5), (13,6), (14,7), (15,2), (16,7), (17,2), (19,2)]

# unitary operator U_B with parameter beta
def U_B(beta):
    for wire in range(n_wires):
        qml.RX(2 * beta, wires=wire)


# unitary operator U_C with parameter gamma
def U_C(gamma):
    for edge in graph:
        wire1 = edge[0]
        wire2 = edge[1]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])


def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)

def rec_qaoao(Hd):
    x=qml.PauliX(Hd)
    Hd=qml.Hadamard(x)
    Hd=qml.PauliX(Hd)


dev = qml.device("default.qubit", wires=n_wires, shots=1)

@qml.qnode(dev)
def circuit(gammas, betas, edge=None, n_layers=1):
    # apply Hadamards to get the n qubit |+> state
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for i in range(n_layers):
        U_C(gammas[i])
        U_B(betas[i])
    if edge is None:
        # measurement phase
        return qml.sample()
    # during the optimization phase we are evaluating a term
    # in the objective using expval
    Hd = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
    #a=qml.expval(Hd)
    #print(a)
    Hd=qml.PauliX(edge[0]) @ qml.PauliX(edge[1])
    #Hd=qml.Hadamard(x)
    #H=qml.PauliX(Hd)
    return qml.expval(Hd)


def qaoa_maxcut(n_layers=1):
    print("\np={:d}".format(n_layers))
    
    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)
    
    # minimize the negative of the objective function
    def objective(params):
        gammas = params[0]
        betas = params[1]
        neg_obj = 0
        for edge in graph:
            print(edge)
            # objective for the MaxCut problem
            neg_obj -= 0.5 * (1 - circuit(gammas, betas, edge=edge, n_layers=n_layers))
        return neg_obj
    
    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)
    
    # optimize parameters in objective
    params = init_params
    steps = 30
    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))
    
    # sample measured bitstrings 100 times
    bit_strings = []
    n_samples = 100
    for i in range(0, n_samples):
        print(bitstring_to_int(circuit(params[0], params[1], edge=None, n_layers=n_layers)))
        bit_strings.append(bitstring_to_int(circuit(params[0], params[1], edge=None, n_layers=n_layers)))
    
    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :n_layers]))
    print("Most frequently sampled bit string is: {:04b}".format(most_freq_bit_string))
    
    return -objective(params), bit_strings


# perform qaoa on our graph with p=1,2 and
# keep the bitstring sample lists
#bitstrings1 = qaoa_maxcut(n_layers=1)[1]
params,bitstrings2 = qaoa_maxcut(n_layers=1)[1]
#bitstrings2 = qaoa_maxcut(n_layers=3)[1]

import matplotlib.pyplot as plt

xticks = range(0, 16)
xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
bins = np.arange(0, 17) - 0.5

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
#plt.subplot(1, 2, 1)
#plt.title("n_layers=1")
#plt.xlabel("bitstrings")
#plt.ylabel("freq.")
#plt.xticks(xticks, xtick_labels, rotation="vertical")
#plt.hist(bitstrings1, bins=bins)
plt.subplot(1, 2, 2)
plt.title("n_layers=2")
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(bitstrings2, bins=bins)
plt.tight_layout()
plt.show()