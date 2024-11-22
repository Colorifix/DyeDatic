import json, subprocess, os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

with open("hyperopt/best_param.json", "r") as f:
    params = json.load(f)

for fold in [0, 1, 2, 4]: # range(5): fold_3 is not used as it is undertrained
    predict_command = f"""chemprop predict \
                          --test-path data/test_natural.csv \
                          --smiles-columns smiles solvent \
                          --model-path hyperopt/best_params/fold_{fold}/model_0/model_v2.pt \
                          --multi-hot-atom-featurizer-mode v1 \
                          --accelerator cpu \
                          --devices auto \
                          --add-h \
                          --preds-path data/preds_natural_fold{fold}_{params["id"]}.csv"""


    process = subprocess.Popen(predict_command.split(),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               )


    stdout, stderr = process.communicate()
    print(stderr.decode("utf-8"))


