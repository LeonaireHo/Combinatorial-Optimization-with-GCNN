import ecole
import shutil
import os
import sys
import time
import numpy as np
import pyscipopt as pyscip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import init

from utility import *
from GCN import *

def getInput(scip_parameters,path):
    env = ecole.environment.Branching(\
                observation_function = ecole.observation.NodeBipartite(),scip_params = scip_parameters)
    instance = ecole.scip.Model.from_file(path)
    obs, _, _, _, _ = env.reset(instance)
#     print(obs)
    
    row_features = np.array(obs.row_features.tolist())
    
    original_indice = obs.variable_features[:, -1]
    l = len(original_indice)
    dict_indice = np.zeros((l))
    for i in range(l):
        dict_indice[int(original_indice[i])] = i
    for i in range(l):
        obs.variable_features[i][-1] = dict_indice[i]
    value = obs.variable_features[obs.variable_features[:, -1].argsort()]
    variable_features = np.array(value[:, :-1].tolist())
    
    for i in range(len(obs.edge_features.indices[1])):
        obs.edge_features.indices[1][i] = dict_indice[obs.edge_features.indices[1][i]]

    value_edge_features = np.array(obs.edge_features.values.tolist())
    indice_edge_features = np.array(obs.edge_features.indices.tolist())
#     print(value_edge_features,indice_edge_features)

    nb_cons = row_features.shape[0]
    nb_var = variable_features.shape[0]
    data_cons = np.hstack((np.zeros((nb_cons,19)),row_features))
    data_var = np.hstack((variable_features,np.zeros((nb_var,5))))
    matrix_H = np.vstack((data_var,data_cons))
#     print(matrix_H)

    matrix_A = np.identity(nb_cons+nb_var)
    for i in range(len(value_edge_features)):
        iVar,iCons = indice_edge_features[1][i],indice_edge_features[0][i] 
        matrix_A[iVar][nb_var+iCons] = value_edge_features[i]
        matrix_A[iCons+nb_var][iVar] = value_edge_features[i]
#     print(matrix_A)
    return [matrix_H,matrix_A,nb_var]


def evaluate(model,H,A,nb_var,scip_parameters,with_diving = False):
    if with_diving:
        predictions = model(torch.from_numpy(H).float(),torch.from_numpy(A).float())[0].squeeze(dim=-1)[:nb_var]
    else:
        predictions = model(torch.from_numpy(H).float(),torch.from_numpy(A).float()).squeeze(dim=-1)[:nb_var]
    aux= torch.Tensor([0.5])
    y_hat = (predictions > aux).float() * 1
    return y_hat,predictions

def heuristique(score,A):
    size = len(score)
    size_cons = A.shape[0] - size
    A = A[:size,size_cons:]
    solution = np.zeros((size))
    nb_cover_original = [len(np.where(A[i]==0)[0])for i in range(size)]
    while len(A[0]) > 0:
        nb_cover = [len(np.where(A[i]!=0)[0]) for i in range(size)]
        i_max = 0
        v_max = 0
        for i,v in enumerate(score):
            aux = v * nb_cover[i] / nb_cover_original[i]
            if aux > v_max :
                i_max = i
                v_max = aux
        solution[i_max] = 1
        for i_cons in np.where(A[i_max]!=0)[0][::-1]:
            A = np.delete(A, i_cons, axis=1)
    return solution

def optimize(scip_parameters,model,lp_file):
    H,A,nb_var = getInput(scip_parameters,lp_file)
    y_hat,predictions = evaluate(model,H,A,nb_var,scip_parameters)
    res_prect = heuristique(predictions.detach().numpy(),A)
    return res_prect,y_hat,A

if __name__ == "__main__":
    scip_parameters = {"branching/scorefunc": "s",
                       "separating/maxrounds": 0,
                       "limits/time": 360,
                       "conflict/enable": False,
                       "presolving/maxrounds": 0,
                       "presolving/maxrestarts": 0,
                       "separating/maxroundsroot" : 0,
                       "separating/maxcutsroot": 0,
                       "separating/maxcuts": 0,
                       "propagating/maxroundsroot" : 0,
                       "lp/presolving" : False,
                      }
    lp = sys.argv[1]
    model_path = sys.argv[2]
    criterion = nn.BCELoss()
    net = VariablePredictor(24,50,3)
    net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    res_her,res_gnn,A = optimize(scip_parameters,net,lp)
    print("Solution by heristirque:",np.where(res_her == 1)[0])
    print("Solution by GNN:",np.where(res_gnn == 1)[0])
    
    
    
    
    
    
    
    
    
    
    
    
