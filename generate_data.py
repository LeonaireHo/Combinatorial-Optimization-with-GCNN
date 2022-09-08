import ecole
import shutil
import os
import sys
import time
import itertools
import numpy as np
import pyscipopt as pyscip
import pickle as pkl
from tqdm import trange

from utility import *
def generate_dataset(scip_parameters,path = "DataSet/",nb_cons = [500],nb_var = [500],density = [0.2],nb_instance = 100,only_problem = False):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    env = ecole.environment.Branching(
                observation_function = ecole.observation.NodeBipartite(),
                scip_params = scip_parameters
                )

    gapList = []

    for row, col, d in itertools.product(nb_cons, nb_var, density):
        setCover = ecole.instance.SetCoverGenerator(n_rows = row, n_cols = col,density = d) 
        print("Generate with Row:%d,Col:%d,Density:%f" % (row,col,d))
        for n in trange(1,nb_instance+1):
            problem_name = "set_cover_{"+row.__str__()+"*"+col.__str__()+"_"+d.__str__()+"_"+n.__str__()+"}"

            done = False
            while(not done):
                try:
                    instance = next(setCover)
                except Exception as e:
                    print(f"Exception occured : {e}")
                    done = True

                if only_problem:
                    instance.write_problem(path+problem_name+".lp")
                else:
                    #save problm lp
                    os.mkdir(path+problem_name)
                    instance.write_problem(path+problem_name+"/problem.lp")

                    #save features
                    obs, _, _, _, _ = env.reset(instance)
                    if obs is None:
                        done = True
                        continue
                    if obs.row_features.shape[0] != row:
                        done = True
                        if os.path.exists(path+problem_name):
                            print(f"Removing : {path+problem_name}")
                            shutil.rmtree(path+problem_name)

                        continue
                    #print("created pb")
                    #save constraintes features
                    dumpRowFeatures(path+problem_name+"/constraints_features.json",obs.row_features)
                    #save variables features
                    dumpVariableFeatures(path+problem_name+"/variables_features.json",obs.variable_features)
                    #save edges features
                    original_indice = obs.variable_features[:,-1]
                    dumpEdgeFeatures(path+problem_name+"/edges_features.json",obs.edge_features,original_indice)
                    #get et save label
                    #print("saved features")
                    solver = ecole.scip.Model.from_file(path+problem_name+"/problem.lp")
                    aspyscip = solver.as_pyscipopt()
                    aspyscip.setPresolve(pyscip.SCIP_PARAMSETTING.OFF)
                    aspyscip.optimize()
                    gapList.append(aspyscip.getGap())
                    dumpSolution_Ecole(path+problem_name+"/label.json",aspyscip)
                    
                done = True             

                #except Exception as ex:
                #    print(row, col, d, "Erreur:%s"%ex)
                #    done = True#False
                #    if os.path.exists(path+problem_name):
                #        print(f"Removing : {path+problem_name}")
                #        shutil.rmtree(path+problem_name)
    gap = np.array(gapList)
    return np.mean(gap)

if __name__=="__main__":
    scip_params = {
        "branching/scorefunc": "s",
        "separating/maxrounds": 0,
        "limits/time": 360,
        "conflict/enable": False,
        "presolving/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "separating/maxroundsroot": 0,
        "separating/maxcutsroot": 0,
        "separating/maxcuts": 0,
        "propagating/maxroundsroot": 0,
        "lp/presolving": False,
    }

    generate_dataset(scip_params, 
            path = "DataSet/",
            nb_cons=[100,200,300,400,500], 
            nb_var=[100,200,300], 
            density=[0.1,0.15,0.2]
            )
