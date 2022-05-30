from models.ESN_clean import ESN
from models.utils import *

kwargs = new_args_dict()
kwargs['model_name'] = 'ESN_clean'

# Flags when running main are given below:
'''
--reservoir_size: default=1000
--sparsity: default=0.1
--radius: default=0.9
--reg: default=1e-7
--alpha: default=1.0
--seed: default=10
'''

# To evaluate the ESN model across all dynamical systems do:

#esn = ESN(**kwargs)
#eval_all_dyn_syst(esn)

esn = ESN(**kwargs)
eval_all_dyn_syst_best_hyperparams(esn, pts_per_period=15)
