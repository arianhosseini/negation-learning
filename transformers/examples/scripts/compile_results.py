import os
import numpy as np
# import seaborn as sns; sns.set_theme()
import pickle
# kl_ws = [1000, 10000, 50000]
exp_dir = "/workdrive/test/local_exps/after_emnlp/distill_zero"
experiments = os.listdir(exp_dir)
mlm_dict = {j:i for i,j in enumerate(["0.0", "0.2", "0.4", "0.6","0.8", "1.0"])}
ulll_dict = {j:i for i,j in enumerate(["0.0", "0.2", "0.4", "0.6"])}


results = {
    "GoogleRE":np.zeros(shape=(6,4)),
    "GoogleRE_Neg":np.zeros(shape=(6,4)),
    "TREx":np.zeros(shape=(6,4)),
    "TREx_Neg":np.zeros(shape=(6,4)),
    "ConceptNet":np.zeros(shape=(6,4)),
    "ConceptNet_Neg":np.zeros(shape=(6,4)),
    "Squad":np.zeros(shape=(6,4)),
    "Squad_Neg":np.zeros(shape=(6,4)),
}
# for klw in kl_ws:
for experiment in experiments:
    # if experiment.split("KL")[-1] == str(klw):
    mlm = experiment[experiment.find("MLMF")+4:experiment.find("MLMF")+7]
    ulll = experiment[experiment.find("ULLL")+5:experiment.find("ULLL")+8]
    i = mlm_dict[mlm]
    j = ulll_dict[ulll]
    try:
        with open(os.path.join(exp_dir, experiment, "lama_results_agg.txt"), 'r') as res_file:
            for line in res_file.readlines():
                task = line.strip().split()[0]
                score = float(line.strip().split()[1])
                results[task][i,j] = score
    except:
        print("not found", experiment)
pickle.dump(results, open("results_distill_zero_compiled.pkl","wb"))
print(results)
