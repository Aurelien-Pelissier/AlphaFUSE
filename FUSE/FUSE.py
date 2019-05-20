import os, sys
import matplotlib.pyplot as plt
import numpy as np

def main():

    Nrepeat =  5 #Number of time we apply FUSE
      #(Obtained best feature subset after each execution is saved in BigResuts.txt)
    
    Nt = 10000                  #Number of iteration per FUSE execution
    dataset = 'Madelon.data'    #Name of the file containing the dataset
    labels = 'Madelon.labels'   #Name of the file containing the corresponding labels
    k = 20                      #Number of nearest neighbors for reward computation
    reward = 'AUC'              #ACC or AUC
    rseed = 0                   #Seed of random generator



    
    
    if reward == 'AUC':
        bool_reward = 1
    else:
        bool_reward = 1
    
    #re
    try:
        os.remove("BigResults.txt")
    except OSError:
        pass

    for rseed in range( Nrepeat):
        os.system('FUSE.exe ' + str(Nt) + ' ' + dataset + ' ' + labels + ' ' + str(k) + ' ' + str(bool_reward) + ' ' + str(rseed))
   
        
    #Read and plot results from BigResults.txt
    study_FUSE_results(dataset)
    
    
    
    
    
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
    
    plt.plot(scores)
    plt.xlabel("Feature index")
    plt.ylabel("Score (= occurence in best feature set)")
	plt.show()
    
    print("  FUSE was executed %.0f times" % nFUSE)
    print("  FUSE average final reward is %.3f" % acc)
    print("  Average number of features selected is %.3f" % n_wav)
    

if __name__ == "__main__":
    main()
