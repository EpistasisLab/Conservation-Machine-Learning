#######################
# Conservation ML     #
# © 2020 moshe sipper #
# www.moshesipper.com #
#######################

USAGE = 'python cml-bdm.py resdir n_replicates n_models n_runs n_samples n_features n_informative n_classes'

from string import ascii_lowercase
from random import choices
from sys import argv, stdin
from os import makedirs
from os.path import exists
from statistics import mean, stdev
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

def rand_str(n): return ''.join(choices(ascii_lowercase, k=n))

def fprint(fname, s):
    if stdin.isatty(): print(s) # running interactively 
    with open(Path(fname),'a') as f: f.write(s)

def get_args():        
    if len(argv) == 9: 
        resdir, n_replicates, n_models, n_runs, n_samples, n_features, n_informative, n_classes =\
            argv[1]+'/', int(argv[2]), int(argv[3]), int(argv[4]), int(argv[5]), int(argv[6]), int(argv[7]), int(argv[8])
    else: # wrong number of args
        exit('-'*80  + '\n' + 'Incorrect usage: python ' + ' '.join(argv) + '\n' + 'Please use: ' + USAGE + '\n' + '-'*80)                    
    if not exists(resdir): makedirs(resdir)    
    s = 'm' + str(n_models) + '_r' + str(n_runs) + '_s' + str(n_samples) + '_f' +  str(n_features) + '_i' + str(n_informative) + '_c' + str(n_classes)
    fname = resdir + s + '_' + rand_str(6) + '.txt'
    return fname, resdir, n_replicates, n_models, n_runs, n_samples, n_features, n_informative, n_classes

def print_params(fname, n_replicates, n_models, n_runs, n_samples, n_features, n_informative, n_classes):
    fprint(fname,\
        'n_replicates: ' + str(n_replicates) + '\n' +\
        'n_models: ' + str(n_models) + '\n' +\
        'n_runs: ' + str(n_runs) + '\n' +\
        'n_samples: ' + str(n_samples) + '\n' +\
        'n_features: ' + str(n_features) + '\n' +\
        'n_informative: ' + str(n_informative) + '\n' +\
        'n_classes: ' + str(n_classes) + '\n')
        
def ensemble_predict(ensemble, dataset, n_classes=2):
    dslen = len(dataset)  
    finalpred = []
    for i in range(dslen): 
        finalpred.append([0] * n_classes)    
    for e in ensemble:
        pred = e.predict(dataset)            
        for i in range(dslen):
            finalpred[i][int(pred[i])] += 1
    finalpred = [p.index(max(p)) for p in finalpred]        
    return finalpred

# main 
def main():       
    fname, resdir, n_replicates, n_models, n_runs, n_samples, n_features, n_informative, n_classes = get_args()
    print_params(fname, n_replicates, n_models, n_runs, n_samples, n_features, n_informative, n_classes)
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_classes=n_classes, n_redundant=0) 

    allreps_rf = [] # rf scores over all replicates
    allreps_jung = [] # jungle of all models    
    for rep in range(1, n_replicates+1):
        onerep_rf = [] # rf scores over one replicate
        onerep_jung = [] # jungle of all models        
        kf = KFold(n_splits=5, shuffle=True) # 5-fold cross validation
        fold = 1
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]   
            jungle = []
            fold_scores = []
            for run in range(n_runs):
                rf = RandomForestClassifier(n_estimators=n_models)                   
                rf.fit(X_train, y_train)
                pred = rf.predict(X_test)
                runscore = balanced_accuracy_score(y_test, pred)
                fold_scores.append(runscore)
                onerep_rf.append(runscore)
                allreps_rf.append(runscore)
                jungle += rf.estimators_ # add all rf trees to jungle

            pred_jung = ensemble_predict(jungle, X_test, n_classes=n_classes)
            jung_score = balanced_accuracy_score(y_test, pred_jung)
            onerep_jung.append(jung_score)
            allreps_jung.append(jung_score)
           
            # stats one fold
            fprint(fname,'replicate ' + str(rep) + ', fold ' + str(fold) + ', ' +
               'mean rfs: ' + str(round(mean(fold_scores),2)) + ' (' + str(round(stdev(fold_scores),2)) + '), ' +\
               'jungle: ' + str(round(jung_score,2)) + '\n')
            
            fold += 1
        
        # stats one replicate
        fprint(fname, 'replicate ' + str(rep) + ', ' +\
            'mean rfs: ' + str(round(mean(onerep_rf),2)) + ' (' + str(round(stdev(onerep_rf),2)) + '), '+\
            'mean jungles: ' + str(round(mean(onerep_jung),2)) + ' (' + str(round(stdev(onerep_jung),2))  + ')\n')
    
    # done replicate runs, report final stats    
    mean_rfs, std_rfs = round(mean(allreps_rf),2), round(stdev(allreps_rf),2)
    mean_jung, std_jung = round(mean(allreps_jung),2), round(stdev(allreps_jung),2)
    impjung = round(100*(mean_jung - mean_rfs)/mean_rfs, 1)
    fprint(fname, '** summary of experiment:\n' +\
       'mean RFs: ' + str(mean_rfs)  + ' (' + str(std_rfs) + ')' + ', ' +\
       'mean jungles (of size ' + str(len(jungle)) + '): ' + str(mean_jung)  + ' (' + str(std_jung) + ')' + ', improvement: ' + str(impjung) + '%\n\n')
 
##############        
if __name__== "__main__":
  main()
  