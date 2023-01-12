#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:13:24 2023

@author: Optimus
"""

import networkx as nx
import numpy as np

from qiskit import Aer,IBMQ
from qiskit import BasicAer
from qiskit.aqua.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.optimization.algorithms import MinimumEigenOptimizer, RecursiveMinimumEigenOptimizer,SolutionSample,OptimizationResultStatus
from qiskit.optimization import QuadraticProgram

# =============================================================================
# from qiskit import BasicAer,Aer
from qiskit.utils import algorithm_globals, QuantumInstance
# from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
# from qiskit.optimization.algorithms import (
#     MinimumEigenOptimizer,
#     RecursiveMinimumEigenOptimizer,
#     SolutionSample,
#     OptimizationResultStatus,
# )
# from qiskit.optimization import QuadraticProgram
# from qiskit.visualization import plot_histogram
from typing import List, Tuple
# import numpy as np
# import networkx as nx
# 
# =============================================================================

G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])#, 20])
G.add_edges_from([(0, 1)])
G.add_edges_from([(0, 2)])
G.add_edges_from([(1, 3)])
G.add_edges_from([(3, 4)])
G.add_edges_from([(4, 5)])
G.add_edges_from([(5, 6)])
G.add_edges_from([(6, 7)])
G.add_edges_from([(7, 2)])
G.add_edges_from([(8, 1)])
G.add_edges_from([(9, 2)])
G.add_edges_from([(10, 3)])
G.add_edges_from([(11, 4)])
G.add_edges_from([(12, 5)])
G.add_edges_from([(13, 6)])
G.add_edges_from([(14, 7)])
G.add_edges_from([(15, 2)])
G.add_edges_from([(16, 7)])
G.add_edges_from([(17, 2)])
G.add_edges_from([(18, 1)])
G.add_edges_from([(19, 2)])
w = nx.adjacency_matrix(G)
n = G.number_of_nodes()

problem = QuadraticProgram()
# create n binary variables
_ = [problem.binary_var('x{}'.format(i)) for i in range(n)]
linear = w.dot(np.ones(n))
quadratic = -w
problem.maximize(linear=linear, quadratic=quadratic)
print(problem.export_as_lp_string())
rep=10
backend = Aer.get_backend('aer_simulator')
#backend=Aer.get_backend('statevector_simulator')
quantum_instance=QuantumInstance(backend)
#cutoff=1
p=3
val=0
for i in range(rep):
    # Run quantum algorithm QAOA on qasm simulator
    backend = BasicAer.get_backend('statevector_simulator')
    exact_mes = NumPyMinimumEigensolver()
    # using the exact classical numpy minimum eigen solver
    exact_solver = MinimumEigenOptimizer(exact_mes)
    exact_result = exact_solver.solve(problem)
    print("exact: {}".format(exact_result))
    qaoa_mes = QAOA(p=p, quantum_instance=backend)
    qaoa_solver = MinimumEigenOptimizer(qaoa_mes)   # using QAOA
    #result = qaoa_mes.run(quantum_instance)
    cutoff = 20
    rqaoa_solver = RecursiveMinimumEigenOptimizer(min_eigen_optimizer=qaoa_solver, min_num_vars=cutoff,min_num_vars_optimizer=exact_solver)
    result = rqaoa_solver.solve(problem)
    print("rqaoa: {}".format(result))
    #print("rqaoa: {}".format(rqaoa_result))
    #print("Optimal value", result['optimal_value'])
    #val+=result['optimal_value']
#print("------- AVERAGE ----------")
#print("Average value", val/rep)


def get_filtered_samples(
     samples: List[SolutionSample],
     threshold: float = 0,
     allowed_status: Tuple[OptimizationResultStatus] = (OptimizationResultStatus.SUCCESS,),
 ):
     res = []
     for s in samples:
         print(s)
         if s.status in allowed_status and s.probability > threshold:
             res.append(s)
 
     return res
 
 
filtered_samples = get_filtered_samples(
     result.samples, threshold=0.005, allowed_status=(OptimizationResultStatus.SUCCESS,)
 )
 
samples_for_plot = {
     " ".join(f"{result.variables[i].name}={int(v)}" for i, v in enumerate(s.x)): s.probability
     for s in filtered_samples
 }
samples_for_plot
# =============================================================================
