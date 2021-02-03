# ​Conservation Random Forests
# © 2021 moshe sipper
# www.moshesipper.com

USAGE =\
'  python crf.py resdir [clf/reg] n_replicates n_models n_runs n_samples n_features n_informative n_classes \n' +\
'  python crf.py resdir [dataset] n_replicates n_models n_runs'

from string import ascii_lowercase
from random import choices, shuffle, choice
from copy import deepcopy 
from sys import argv, stdin
from os import makedirs
from os.path import exists
from pandas import read_csv
from statistics import mean, stdev
from pathlib import Path
from operator import itemgetter
from sklearn.datasets import make_classification, fetch_openml
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from pmlb import fetch_data, classification_dataset_names
from mlxtend.evaluate import permutation_test
from decimal import Decimal

def rand_str(n): return ''.join(choices(ascii_lowercase, k=n))

def fprint(fname, s):
    if stdin.isatty(): print(s) # running interactively 
    with open(Path(fname),'a') as f: f.write(s)

def get_args():      
    if len(argv) not in [6,10]: # wrong number of args
        exit('-'*80                       + '\n' +\
             'Incorrect usage:'           + '\n' +\
             '  python ' + ' '.join(argv) + '\n' +\
             'Please use:'                + '\n' +\
             USAGE                        + '\n' +\
             '-'*80)
    
    resdir, dsname, n_replicates, n_models, n_runs = argv[1]+'/', argv[2], int(argv[3]), int(argv[4]), int(argv[5])    
    if len(argv) == 10: # for make_classification
        n_samples, n_features, n_informative, n_classes = int(argv[6]), int(argv[7]), int(argv[8]), int(argv[9])
    elif len(argv) == 6: # for all other datasets
        n_samples, n_features, n_informative, n_classes = -1, -1, -1, -1
                    
    if not exists(resdir): makedirs(resdir)
    
    runtype = dsname
    if dsname == 'clf':
        runtype += '_m' + str(n_models) + '_r' + str(n_runs) + '_s' + str(n_samples) + '_f' +  str(n_features) + '_i' +   str(n_informative) + '_c' + str(n_classes)
    fname = resdir + runtype + '_' + rand_str(6) + '.txt'

    return fname, resdir, dsname, n_replicates, n_models, n_runs, n_samples, n_features, n_informative, n_classes

def print_params(fname, dsname, n_replicates, n_models, n_runs, n_samples, n_features, n_informative, n_classes):
    fprint(fname,\
        'dsname:        ' + dsname             + '\n' +\
        'n_replicates:  ' + str(n_replicates)  + '\n' +\
        'n_models:      ' + str(n_models)      + '\n' +\
        'n_runs:        ' + str(n_runs)        + '\n' +\
        'n_samples:     ' + str(n_samples)     + '\n' +\
        'n_features:    ' + str(n_features)    + '\n' +\
        'n_informative: ' + str(n_informative) + '\n' +\
        'n_classes:     ' + str(n_classes)) 

def get_dataset(dsname, n_samples, n_features, n_informative, n_classes):
    if dsname == 'clf':
        X, y = make_classification(n_samples=n_samples, n_features=n_features,\
                  n_informative=n_informative, n_classes=n_classes, n_redundant=0)
    else:
        if dsname == 'cancer':
            X, y = load_breast_cancer(return_X_y=True)
        elif dsname == 'iris':
            X, y = load_iris(return_X_y=True)
        elif dsname == 'wine':
            X, y = load_wine(return_X_y=True)
        elif dsname == 'digits':
            X, y = load_digits(return_X_y=True)
        elif dsname in classification_dataset_names: 
            X, y = fetch_data(dsname, return_X_y=True, local_cache_dir='pmlb')
        else:
            try: # dataset from openml?
                X, y = fetch_openml(dsname, return_X_y=True, as_frame=False, cache=False)
            except:
                try: # a csv file in datasets folder?
                    data = read_csv('datasets/' + dsname + '.csv', sep=',')
                    array = data.values
                    X, y = array[:,0:-1], array[:,-1] # target is last col
                    # X, y = array[:,1:], array[:,0] # target is 1st col
                except Exception as e: 
                    print('looks like there is no such dataset')
                    exit(e)
            
        n_samples, n_features = X.shape
        n_classes = len(set(y))
        n_informative = -1
        if dsname.find("xor") != -1: n_informative = ' '.join(list(data.columns.values)).count('M0')
       
    le = LabelEncoder() # encode target labels with values between 0 and n_classes-1
    y = le.fit(y).transform(y)
                                
    return X, y, n_samples, n_features, n_informative, n_classes

def score(target, pred):
    return balanced_accuracy_score(target, pred)
        
def ensemble_predict(ensemble, X, n_classes=2):
    dslen = len(X)  
    finalpred = []
    for i in range(dslen): 
        finalpred.append([0] * n_classes)
    for e in ensemble:
        pred = e.predict(X)            
        for i in range(dslen):
            finalpred[i][int(pred[i])] += 1
    finalpred = [p.index(max(p)) for p in finalpred]
    return finalpred

def lexicase(population, outvecs, targets): 
# lexicase selection, https://faculty.hampshire.edu/lspector/pubs/lexicase-beyond-gp-preprint.pdf
    if len(targets) != len(outvecs[0]): exit("lexicase error: target length does not match output length")
    if len(population) != len(outvecs): exit("lexicase error: number of models does not match number of output vectors")
    candidates = list(range(len(population)))
    test_cases = list(range(len(targets)))
    shuffle(test_cases)
    while True:
        case = test_cases[0]
        best_on_first_case = [c for c in candidates if outvecs[c][case] == targets[case] ]                              
        if len(best_on_first_case) > 0: candidates = best_on_first_case                
        if len(candidates) == 1: return deepcopy(population[candidates[0]])
        del test_cases[0]
        if len(test_cases) == 0: 
            return deepcopy(population[choice(candidates)])

def lexi_garden(models, outputs, y, n_models=100):
    # construct ensemble through lexigarden algorithm
    garden = []
    for i in range(n_models): 
        garden.append(lexicase(models, outputs, y))
    return garden

def order_based_pruning(models_scores, outputs, n_models=100):
    # construct ensemble from top-scoring models
    # models_scores is a list of [ [model, score],..... ]
    srt = sorted(models_scores, key=itemgetter(1), reverse=True)
    return [m[0] for m in srt][:n_models]

def cluster_based_pruning(models_scores, outputs, n_clusters=10):
    # construct ensemble using cluster-based pruning
    # models_scores is a list of [ [model, score],..... ]
    ensemble = []
    for i in range(n_clusters): 
        ensemble.append([])
    clusters = KMeans(n_clusters=n_clusters).fit_predict(outputs)
    for i, c in enumerate(clusters):
        ensemble[c].append(models_scores[i])
    final = []
    for i in range(n_clusters): 
        # print('\n\n',i, ensemble[i])
        srt = sorted(ensemble[i], key=itemgetter(1), reverse=True)
        if len(srt)>0: final.append(srt[0][0])
    return final

# main 
def main():       
    fname, resdir, dsname, n_replicates, n_models, n_runs, n_samples, n_features, n_informative, n_classes = get_args()
    X, y, n_samples, n_features, n_informative, n_classes = get_dataset(dsname, n_samples, n_features, n_informative, n_classes)
    print_params(fname, dsname, n_replicates, n_models, n_runs, n_samples, n_features, n_informative, n_classes)

    allreps_rf = [] # rf scores, allreps == all replicate runs
    allreps_order300 = [] # order-based pruning, 300 models
    allreps_order1000 = [] # order-based pruning, 1000 models
    allreps_cluster20 = [] # cluster-based pruning, 20 clusters
    allreps_cluster50 = [] # cluster-based pruning, 50 clusters
    allreps_lexi300  = [] # lexigarden of size 300
    allreps_lexi1000 = [] # lexigadren of size 1000
    allreps_super = [] # super-ensemble (ensemble of ensembles)
    allreps_jungle  = [] # jungle of all models
    
    for rep in range(1, n_replicates+1):
        onerep_rf = [] # rf scores, onerep = one replicate run
        onerep_order300 = [] # order-based pruning, 300 models
        onerep_order1000 = [] # order-based pruning, 1000 models
        onerep_cluster20 = [] # cluster-based pruning, 20 clusters
        onerep_cluster50 = [] # cluster-based pruning, 50 clusters
        onerep_lexi300  = [] # lexigarden of size 300
        onerep_lexi1000 = [] # lexigadren of size 1000
        onerep_super = [] # super-ensemble
        onerep_jungle = [] # jungle of all models
        
        kf = KFold(n_splits=5, shuffle=True) # 5-fold cross validation
        fprint(fname, '\n\nn_splits:      ' + str (kf.get_n_splits(X)) + '\n')
        fold = 1
        for train_index, test_index in kf.split(X):
            jungle = []
            super_ensemble = [] 
            fold_scores = []
            for run in range(n_runs):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]   
                rf = RandomForestClassifier(n_estimators=n_models)                    
                rf.fit(X_train, y_train)
                pred = rf.predict(X_test)
                runscore = score(y_test, pred) # score of a single rf
                fold_scores.append(runscore)
                onerep_rf.append(runscore)
                allreps_rf.append(runscore)
                jungle += rf.estimators_
                super_ensemble.append(rf)

            outputs = []
            for m in jungle: 
                outputs.append(m.predict(X_train))            
            models_scores = []
            for m, o in zip(jungle, outputs): 
                models_scores.append([m, score(y_train, o)])
            
            # create the various ensembles
            order300 = order_based_pruning(models_scores, outputs, n_models=300)
            order1000 = order_based_pruning(models_scores, outputs, n_models=1000)
            cluster20 = cluster_based_pruning(models_scores, outputs, n_clusters=20)
            cluster50 = cluster_based_pruning(models_scores, outputs, n_clusters=50)
            lexi300  = lexi_garden(jungle, outputs, y_train, n_models=300)
            lexi1000 = lexi_garden(jungle, outputs, y_train, n_models=1000)
            
            # test the various ensembles on test set
            pred_order300 = ensemble_predict(order300, X_test, n_classes=n_classes)
            pred_order1000 = ensemble_predict(order1000, X_test, n_classes=n_classes)
            pred_cluster20 = ensemble_predict(cluster20, X_test, n_classes=n_classes)
            pred_cluster50 = ensemble_predict(cluster50, X_test, n_classes=n_classes)        
            pred_lexi300 = ensemble_predict(lexi300, X_test, n_classes=n_classes)
            pred_lexi1000 = ensemble_predict(lexi1000, X_test, n_classes=n_classes)
            pred_jungle = ensemble_predict(jungle, X_test, n_classes=n_classes)
            pred_super = ensemble_predict(super_ensemble, X_test, n_classes=n_classes)

            order_score300 = score(y_test, pred_order300)
            onerep_order300.append(order_score300)
            allreps_order300.append(order_score300)

            order_score1000 = score(y_test, pred_order1000)
            onerep_order1000.append(order_score1000)
            allreps_order1000.append(order_score1000)
 
            cluster_score20 = score(y_test, pred_cluster20)
            onerep_cluster20.append(cluster_score20)
            allreps_cluster20.append(cluster_score20)

            cluster_score50 = score(y_test, pred_cluster50)
            onerep_cluster50.append(cluster_score50)
            allreps_cluster50.append(cluster_score50)
           
            lexi_score300 = score(y_test, pred_lexi300)
            onerep_lexi300.append(lexi_score300)
            allreps_lexi300.append(lexi_score300)
            
            lexi_score1000 = score(y_test, pred_lexi1000)            
            onerep_lexi1000.append(lexi_score1000)
            allreps_lexi1000.append(lexi_score1000)

            jungle_score = score(y_test, pred_jungle)
            onerep_jungle.append(jungle_score)
            allreps_jungle.append(jungle_score)

            super_score = score(y_test, pred_super)            
            onerep_super.append(super_score)
            allreps_super.append(super_score)
            
           
            # stats one fold
            fprint(fname,'replicate ' + str(rep) + ', fold ' + str(fold) + ', ' +\
               'mean rfs: ' +\
                str(round(mean(fold_scores),2)) + ' (' + str(round(stdev(fold_scores),2)) + '), ' +\
                'means others: ' +\
                str(round(order_score300,2)) + ', ' + str(round(order_score1000,2)) + ', ' +\
                str(round(cluster_score20,2)) + ', ' + str(round(cluster_score50,2)) + ', ' +\
                str(round(lexi_score300,2)) + ', ' + str(round(lexi_score1000,2)) + ', ' +\
                str(round(jungle_score,2))     + ', ' + str(round(super_score,2)) + '\n')
            
            fold += 1
        
        
        # stats one replicate
        fprint(fname, 'replicate ' + str(rep) + ', ' +\
            'mean rfs: ' +\
            str(round(mean(onerep_rf),2)) + ' (' + str(round(stdev(onerep_rf),2)) + '), '+\
            'means others: ' +\
            str(round(mean(onerep_order300),2)) + ' (' + str(round(stdev(onerep_order300),2)) + '), ' +\
            str(round(mean(onerep_order1000),2)) + ' (' + str(round(stdev(onerep_order1000),2)) + '), ' +\
            str(round(mean(onerep_cluster20),2)) + ' (' + str(round(stdev(onerep_cluster20),2)) + '), ' +\
            str(round(mean(onerep_cluster50),2)) + ' (' + str(round(stdev(onerep_cluster50),2)) + '), ' +\
            str(round(mean(onerep_lexi300),2)) + ' (' + str(round(stdev(onerep_lexi300),2)) + '), ' +\
            str(round(mean(onerep_lexi1000),2)) + ' (' + str(round(stdev(onerep_lexi1000),2)) + '), ' +\
            str(round(mean(onerep_jungle),2)) + ' (' + str(round(stdev(onerep_jungle),2)) + '), ' +\
            str(round(mean(onerep_super),2)) + ' (' + str(round(stdev(onerep_super),2)) + ')\n'  )

    # done replicate runs, compute and report final stats
    
    # mean, stddev
    mean_rf, std_rf = mean(allreps_rf), stdev(allreps_rf)
    mean_order300, std_order300 = mean(allreps_order300), stdev(allreps_order300)
    mean_order1000, std_order1000 = mean(allreps_order1000), stdev(allreps_order1000)
    mean_cluster20, std_cluster20 = mean(allreps_cluster20), stdev(allreps_cluster20)
    mean_cluster50, std_cluster50 = mean(allreps_cluster50), stdev(allreps_cluster50)
    mean_lexi300, std_lexi300  = mean(allreps_lexi300), stdev(allreps_lexi300)
    mean_lexi1000, std_lexi1000 = mean(allreps_lexi1000), stdev(allreps_lexi1000)
    mean_jungle, std_jungle = mean(allreps_jungle), stdev(allreps_jungle)
    mean_super, std_super   = mean(allreps_super), stdev(allreps_super)
    
    # percent improvment
    imp_order300 = round(100*(mean_order300 - mean_rf)/mean_rf, 1)
    imp_order1000 = round(100*(mean_order1000 - mean_rf)/mean_rf, 1)
    imp_cluster20 = round(100*(mean_cluster20 - mean_rf)/mean_rf, 1)
    imp_cluster50 = round(100*(mean_cluster50 - mean_rf)/mean_rf, 1)
    imp_lexi300  = round(100*(mean_lexi300 - mean_rf)/mean_rf, 1)
    imp_lexi1000 = round(100*(mean_lexi1000 - mean_rf)/mean_rf, 1)
    imp_jungle = round(100*(mean_jungle - mean_rf)/mean_rf, 1)
    imp_super = round(100*(mean_super - mean_rf)/mean_rf, 1)
    
    # p-value of permutation test
    rounds = 10000 # number of permutation-test rounds
    pval_order300 = permutation_test(allreps_rf, allreps_order300, method='approximate', num_rounds=rounds)
    pval_order1000 = permutation_test(allreps_rf, allreps_order1000, method='approximate', num_rounds=rounds)
    pval_cluster20 = permutation_test(allreps_rf, allreps_cluster20, method='approximate', num_rounds=rounds)
    pval_cluster50 = permutation_test(allreps_rf, allreps_cluster50, method='approximate', num_rounds=rounds)
    pval_lexi300  = permutation_test(allreps_rf, allreps_lexi300, method='approximate', num_rounds=rounds)
    pval_lexi1000 = permutation_test(allreps_rf, allreps_lexi1000, method='approximate', num_rounds=rounds)
    pval_jungle = permutation_test(allreps_rf, allreps_jungle, method='approximate', num_rounds=rounds)
    pval_super  = permutation_test(allreps_rf, allreps_super, method='approximate', num_rounds=rounds)
    
    # for marking whether pval <= 0.05 and/or <= 0.001
    th1, th2 = 0.001, 0.05
    pord300 = ' (!!)' if pval_order300 <th1 else ' (!)' if pval_order300 <th2 else ''
    pord1000 = ' (!!)' if pval_order1000 <th1 else ' (!)' if pval_order1000 <th2 else ''
    pclu20  = ' (!!)' if pval_cluster20 <th1 else ' (!)' if pval_cluster20 <th2 else ''
    pclu50  = ' (!!)' if pval_cluster50 <th1 else ' (!)' if pval_cluster50 <th2 else ''
    plex300  = ' (!!)' if pval_lexi300 <th1 else ' (!)' if pval_lexi300 <th2 else ''
    plex1000 = ' (!!)' if pval_lexi1000<th1 else ' (!)' if pval_lexi1000<th2 else ''
    pjung = ' (!!)' if pval_jungle<th1 else ' (!)' if pval_jungle<th2 else ''
    psup  = ' (!!)' if pval_super <th1 else ' (!)' if pval_super <th2 else ''
            
    fprint(fname, '\n>> dataset & n_models & n_runs & n_samples & n_features & n_informative & n_classes & mean_rf (sd) & size & mean_ord300 (sd) & %imp, pval_ & mean_ord1000 (sd) & %imp, pval_ & mean_clus300 (sd) & %imp, pval_ & mean_clus1000 (sd) & %imp, pval_ & mean_lexi300 (sd) & %imp, pval_  & size & mean_lexi1000 (sd) & %imp, pval_  & size & mean_jungle (sd) & %imp, pval_ & size & mean_super (sd) & %imp, pval_ \n\n')
        
    fprint(fname, '>> ' +\
       dsname + ' & '  + str(n_models) + ' & ' + str(n_runs) + ' & ' + str(n_samples) + ' & ' +\
       str(n_features) + ' & ' + str(n_informative) + ' & ' + str(n_classes) + ' & ' +\
       str(round(mean_rf),2) + ' ('  + str(round(std_rf),2) + ')' + ' & ' +\
       str(len(order300)) + 'm & ' + str(mean_order300)     + ' (' + str(std_order300) + ')' + ' & ' +\
       str(imp_order300) + '\\%, '   + '%.1E' % Decimal(pval_order300)  + pord300 + ' & ' +\
       str(len(order1000)) + 'm & ' + str(mean_order1000)     + ' (' + str(std_order1000) + ')' + ' & ' +\
       str(imp_order1000) + '\\%, '   + '%.1E' % Decimal(pval_order1000)  + pord1000 + ' & ' +\
       str(len(cluster20)) + 'm & ' + str(mean_cluster20)     + ' (' + str(std_cluster20) + ')' + ' & ' +\
       str(imp_cluster20) + '\\%, '   + '%.1E' % Decimal(pval_cluster20)  + pclu20 + ' & ' +\
       str(len(cluster50)) + 'm & ' + str(mean_cluster50)     + ' (' + str(std_cluster50) + ')' + ' & ' +\
       str(imp_cluster50) + '\\%, '   + '%.1E' % Decimal(pval_cluster50)  + pclu50 + ' & ' +\
       str(len(lexi300)) + 'm & ' + str(mean_lexi300)     + ' (' + str(std_lexi300) + ')' + ' & ' +\
       str(imp_lexi300) + '\\%, ' + '%.1E' % Decimal(pval_lexi300)  + plex300 + ' & ' +\
       str(len(lexi1000)) + 'm & ' + str(mean_lexi1000)    + ' (' + str(std_lexi1000) + ')' + ' & ' +\
       str(imp_lexi1000) + '\\%, ' + '%.1E' % Decimal(pval_lexi1000) + plex1000 + ' & ' +\
       str(len(jungle)) + 'm & ' + str(mean_jungle)    + ' (' + str(std_jungle) + ')' + ' & ' +\
       str(imp_jungle) + '\\%, ' + '%.1E' % Decimal(pval_jungle) + pjung + ' & ' +\
       str(len(super_ensemble)) + 'e & ' + str(mean_super) + ' (' + str(std_super) + ')' + ' & ' +\
       str(imp_super) + '\\%, ' + '%.1E' % Decimal(pval_super) + psup + ' \\\\' +\
       '\n\n')

    inf = '---' if n_informative == -1 else str(n_informative)
    fprint(fname, '** ' +\
       dsname+ ' & '+ str(n_samples) + ' & '+ str(n_features) + ' & '+ inf + ' & '+ str(n_classes)+ ' & ' +\
       str(round(mean_rf),2)  + ' (' + str(round(std_rf),2) + ')' + ' & ' +\
       str(imp_jungle) + '\\%, ' + '%.1E' % Decimal(pval_jungle) + pjung + ' & ' +\
       str(imp_super) + '\\%, ' + '%.1E' % Decimal(pval_super) + psup + ' & ' +\
       str(imp_order300) + '\\%, ' + '%.1E' % Decimal(pval_order300) + pord300 + ' & ' +\
       str(imp_order1000) + '\\%, ' + '%.1E' % Decimal(pval_order1000) + pord1000 + ' & ' +\
       str(imp_cluster20) + '\\%, ' + '%.1E' % Decimal(pval_cluster20) + pclu20 + ' & ' +\
       str(imp_cluster50) + '\\%, ' + '%.1E' % Decimal(pval_cluster50) + pclu50 + ' & ' +\
       str(imp_lexi300) + '\\%, ' + '%.1E' % Decimal(pval_lexi300) + plex300 + ' & ' +\
       str(imp_lexi1000) + '\\%, ' + '%.1E' % Decimal(pval_lexi1000) + plex1000  +\
       ' \\\\' + '\n\n')

'''                   
    # sanity check -- compare permutation testing with Welch's (even though latter assumes independence)
    w300  = stats.ttest_ind(allreps_rf, allreps_lexi300,  equal_var = False)[1]
    w1000 = stats.ttest_ind(allreps_rf, allreps_lexi1000, equal_var = False)[1]
    wjung = stats.ttest_ind(allreps_rf, allreps_jungle,       equal_var = False)[1]
    wsup = stats.ttest_ind(allreps_rf, allreps_super,         equal_var = False)[1]
    
    fprint(fname, '>> Welch: ' +\
    '%.1E' % Decimal(pval_lexi1000) +  ', ' + '%.1E' % Decimal(w1000) + ', ' + '%.1E' % Decimal(abs(pval_lexi1000-w1000)) + ' | ' +\
    '%.1E' % Decimal(pval_jungle) +  ', ' + '%.1E' % Decimal(wjung) + ', ' + '%.1E' % Decimal(abs(pval_jungle-wjung)) + ' | ' +\
    '%.1E' % Decimal(pval_super)  +  ', ' + '%.1E' % Decimal(wsup)  + ', ' + '%.1E' % Decimal(abs(pval_super-wsup)) + '\n\n')
'''

##############        
if __name__== "__main__":
  main()
  
