import os, sys
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    
    print("\n") 
    print("\n") 
    print("==================================================================================") 
    print("=================================== Alpha-FUSE ===================================") 
    print("==================================================================================\n") 
    
    # An accurate simulation would need at least Nrepeat = 100 and Nt = 200000, but involve overnight computation
    # Running with boot_frac = 1, Nrepeat = 1 and Nt = 200000 can already give you a good feature set.
      

    Nrepeat =  1   #Number of time we apply FUSE (results will be averaged)
      #(Obtained best feature subset after each execution is saved in BigResuts.txt)
    Nt = 100000                  #Number of iteration per FUSE execution
    boot_frac = 0.75             #Number of example to consider at each FUSE execution (relatively to the number of total example) - usefull to avoid overfitting
    dataset = 'Madelon.data'     #Name of the file containing the dataset
    labels = 'Madelon.labels'    #Name of the file containing the corresponding labels
    k = 5                        #Number of nearest neighbors for reward computation
    reward = 'ACC'               #ACC for accuracy or AUC
    if reward == 'AUC':
        bool_reward = 1
    else:
        bool_reward = 0
      
    


    try:
        os.remove("BigResults.txt") #File containign the results of FUSE simulation
    except OSError:
        pass
    
    #launch alphaFUSE
    #let the the simulation run (can take time)
    print("  Running %s FUSE simulations with %s iterations each ..." % (Nrepeat,Nt))
    os.system('Feature_Selection.exe ' + str(boot_frac) + ' ' + str(Nt) + ' ' + dataset + ' ' + labels + ' ' + str(k) + ' ' + str(bool_reward) + ' ' + str(Nrepeat))
        
    #Get the Results     
    scores = study_FUSE_results(dataset)
    np.save("FUSEscores",scores)
    
    rank = np.argsort(-scores)
    print("\n")
    for i in range(5):
        print("   %sth score is feature %s with a score of %.2f " % (i+1, rank[i], scores[rank[i]]))
    
        
        
    
    
    
def study_FUSE_results(dataset):
    
    data = np.loadtxt(dataset)
    nFeatures = data.shape[1]
    scores = np.zeros(nFeatures)
    nFUSE = 0
    
    result_file = "BigResults.txt"
    
    try:
        with open(result_file):
            print("\n")
    except IOError:
        print("\n  BigResults.txt does not exist")
        sys.exit(0)
    
    text_file = open(result_file, "r")
    lines = text_file.read().split(';')
    acc = 0
    n_wav = 0
    
    for element in lines:
        if "[" in element:
            nFUSE += 1
            n_features = 0
            for i in range(nFeatures):
                if str(i).zfill(3) in element:
                    n_features += 1
                    scores[i] += 1
            n_wav += n_features
        elif "n" not in element and "0" in element:
            acc+= float(element)
    
    
    text_file.close()
    scores /= nFUSE
    acc /= nFUSE
    n_wav /= nFUSE
    
    plt.figure(figsize=(7,4))
    n, bins, patches = plt.hist(np.arange(len(scores)),bins=len(scores), weights=scores,alpha=0.7, rwidth=0.85, color="red")
    plt.xlabel("Feature index")
    plt.ylabel("Score (= occurence in best feature set)")
    plt.show()

    print("  FUSE was executed %.0f times" % nFUSE)
    print("  FUSE average final reward is %.3f" % acc)
    print("  Average number of features selected is %.3f" % n_wav)
    
    return scores
    

if __name__ == "__main__":
    main()
