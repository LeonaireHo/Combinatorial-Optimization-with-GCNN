# Combinatorial-Optimization-with-GCNN
## Install ecole
The part of this project to get features depends on library [ecole](https://markdown.com.cn).But we made some changes in ecole, so you need to install our version by :https://github.com/LeonaireHo/ecole.
The environment required to build ecole through conda：
~~~
conda env create -n ecole -f dev/conda.yaml
~~~
In the Ecole source repository,install with setup.py
~~~
python setup.py install
~~~

## Generate of Dataset
~~~python
from generate_data import *

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

generate_dataset(scip_parameters,path = "DataSet/",\
                 nb_cons = [100,200,300,400,500],nb_var = [1,1.5,2],density = [0.1,0.15,0.2])
~~~
