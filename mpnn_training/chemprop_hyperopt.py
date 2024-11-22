# A hyperparameter optimisation of MPNN model using approach
# from Greenman et al. https://doi.org/10.1039/D1SC05677H
# original training data was cleaned and enriched with natural colourants data
# training goes on GPU the chemprop must be configured accordingly

import numpy as np
from hyperopt import hp, fmin, tpe, Trials, space_eval, STATUS_OK
import json, os, subprocess, uuid

SAVE_DIR = "hyperopt"

def objective_fun(params):
    param_id = str(uuid.uuid4())
    params["id"] = param_id

    # make sure max learning rate is bigger than init_lr and final_lr
    params['init-lr'] = params['init-lr'] * params['max-lr']
    params['final-lr'] = params['final-lr'] * params['max-lr']

    trial_directory = os.path.join(SAVE_DIR, param_id)
    hyperopt_config_dir = os.path.join(trial_directory, "hyperopt.json")
    if not os.path.exists(trial_directory):
        os.makedirs(trial_directory)

    # keep the parameters saved
    with open(hyperopt_config_dir, "w") as outfile:
        json.dump(params, outfile)

    run_command = f"""chemprop train \
                     --data-path data/data_all.csv \
                     --smiles-columns smiles solvent \
                     --task-type regression \
                     --target-columns peakwavs_max \
                     --loss-function mse \
                     --splits-column split \
                     --data-seed 123 \
                     --pytorch-seed 42 \
                     --metric mae rmse \
                     --save-dir {trial_directory} \
                     --batch-size {params["batch-size"]} \
                     --message-hidden-dim {params["message-hidden-dim"]} \
                     --activation {params["activation"]} \
                     --aggregation {params["aggregation"]} \
                     --depth {params["depth"]} \
                     --dropout {params["dropout"]} \
                     --ffn-num-layers {params["ffn-num-layers"]} \
                     --ffn-hidden-dim {params["ffn-hidden-dim"]} \
                     --warmup-epochs {params["warmup-epochs"]} \
                     --init-lr {params["init-lr"]} \
                     --max-lr {params["max-lr"]} \
                     --final-lr {params["final-lr"]} \
                     --add-h \
                     --epochs 30 \
                     --accelerator cpu \
                     --devices auto \
                     --multi-hot-atom-featurizer-mode v1 \
                     {params["message-bias"]} \
                     --ensemble-size 1"""

    process = subprocess.Popen(run_command.split(),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               )
    
    stdout, stderr = process.communicate()
    print(stderr)
    results_file = os.path.join(trial_directory, "verbose.log")
    with open(results_file, 'r') as f:
        results = f.readlines()
    final_val_mae = np.mean([float(y.split()[6]) for y in [x for x in results if 'best validation mae' in x]])

    return {'loss': final_val_mae, 'status': STATUS_OK}


param_space = {
    'message-bias': hp.choice('message-bias', ['--message-bias', '']),
    'message-hidden-dim': hp.choice('message-hidden-dim', range(100, 420, 20)),
    'activation': hp.choice('activation', ['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU']),
    'aggregation': hp.choice('aggregation', ['mean', 'sum', 'norm']),
    'depth': hp.choice('depth', [3, 4, 5, 6]),
    'dropout': hp.uniform('dropout', 0.0, 0.4),
    'ffn-num-layers': hp.choice('ffn-num-layers', [1, 2, 3]),
    'ffn-hidden-dim': hp.choice('ffn-hidden-dim', range(100, 420, 20)),
    'warmup-epochs': hp.choice('warmup-epochs', [2, 3, 4, 5, 6]),
    'batch-size': hp.choice('batch-size', range(10, 110, 20)),
    'init-lr': hp.loguniform('init-lr', -3, -1),
    'max-lr': hp.loguniform('max-lr', -5, -2),
    'final-lr': hp.loguniform('final-lr', -3, -1)}

trials = Trials()

best_params = fmin(
    fn=objective_fun,
    space=param_space,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials)

with open(best_params, "w") as outfile:
    json.dump(params, outfile)
