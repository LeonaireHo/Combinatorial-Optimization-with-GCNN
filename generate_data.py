import ecole
import shutil
import os
import sys
import time
import numpy as np
import pyscipopt as pyscip
import pickle as pkl
from tqdm import trange

from utility import *
def generate_dataset(scip_parameters,path = "DataSet/",nb_cons = [500],nb_var = [500],density = [0.2],nb_instance = 100,only_problem = False):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    env = ecole.environment.Branching(\
                observation_function = ecole.observation.NodeBipartite(),scip_params = scip_parameters)
    gapList = []
    for row in nb_cons:
        for col in nb_var: 
            for d in density:
                setCover = ecole.instance.SetCoverGenerator(n_rows = row, n_cols = col,density = d) 
                print("Generate with Row:%d,Col:%d,Density:%f" % (row,col,d))
                for n in trange(1,nb_instance+1):
                    done = False
                    while(not done):
                        try:
                            problem_name = \
                                "set_cover_{"+row.__str__()+"*"+col.__str__()+"_"+d.__str__()+"_"+n.__str__()+"}"
                            instance = next(setCover)
                            if only_problem:
                                #save problm lp
                                problem_name = \
                                    "set_cover_{"+row.__str__()+"*"+col.__str__()+"_"+d.__str__()+"_"+n.__str__()+"}"
                                instance.write_problem(path+problem_name+".lp")
                            else:
                                #save problm lp
                                problem_name = \
                                    "set_cover_{"+row.__str__()+"*"+col.__str__()+"_"+d.__str__()+"_"+n.__str__()+"}"
                                os.mkdir(path+problem_name)
                                instance.write_problem(path+problem_name+"/problem.lp")
                                #save features
                                obs, _, _, _, _ = env.reset(instance)
                                if obs.row_features.shape[0] != row:
                                    raise Exception
                                print("created pb")
                                #save constraintes features
                                dumpRowFeatures(path+problem_name+"/constraints_features.json",obs.row_features)
                                #save variables features
                                dumpVariableFeatures(path+problem_name+"/variables_features.json",obs.variable_features)
                                #save edges features
                                original_indice = obs.variable_features[:,-1]
                                dumpEdgeFeatures(path+problem_name+"/edges_features.json",obs.edge_features,original_indice)
                                #get et save label
                                print("saved features")
                                solver = ecole.scip.Model.from_file(path+problem_name+"/problem.lp")
                                aspyscip = solver.as_pyscipopt()
                                aspyscip.setPresolve(pyscip.SCIP_PARAMSETTING.OFF)
                                aspyscip.optimize()
                                gapList.append(aspyscip.getGap())
                                dumpSolution_Ecole(path+problem_name+"/label.json",aspyscip)
                            
                            done = True             
                        except Exception as ex:
                            print("Erreur:%s"%ex)
                            done = False
                            shutil.rmtree(path+problem_name)
    gap = np.array(gapList)
    return np.mean(gap)
