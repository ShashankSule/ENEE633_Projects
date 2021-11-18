Welcome to my documentation for Project 1! My code is organized by the two problems on the project. 
All the code is in MATLAB so you first must have MATLAB open. 


# Problem 1 

1. Bayes' Classification--Run the file `BayesClassification_illum.m` in MATLAB. Simple! 
2. k-NN classfication--Run the file '`kNN_illumination.m` in MATLAB. Wow, easy. 
3. Loading data--The file `loadandfiddleIllumination.m` loads the illumination dataset to the workspace, organizes the flattened subject-wise into different 
68 classes stored as a MATLAB cell `classes`. If you want to do this process sequentially or tweak the number of dimensions in PCA or MDA, you can comment
out the first and last lines of the code. 

# Problem 2 

1. Bayes' Classification--Run the file `BayesClassification_DATA.m`.
2. k-NN classification--Run the file `kNN.m`. 
3. SVM--First make sure to put upload `opt_r_MDA.mat` and `opt_sigma.mat` (found in the "Data" subdirectory) into your variable workspace on MATLAB. Next, go open the file `svm.m` and comment out the relevant kernel on lines 28-30 and the relevant parameter on lines 26-27 (for example, you should comment out lines 26 and 28 to generate testing-error statistics for RBF-kernel SVN using optimal variance parameters). If you would like to run AdaBoost on these classifiers, simply uncomment line #29. This code will generate the error vector `testing_err` which you can write to file if you so choose. It will also generate the labels for the testing set stored in `test` on line 24. 
4. Cross-validation--Run the file `crossvalidation.m`! 
5. WeakClassifier--This module picks a weak classifier using the following strategy: (a) First, multiply each the label vector with the probability vector $p$ and solve the dual SVM problem to a low number of iterations (say 100)--this "tunes" the SVM problem to the high-signficance data points as quantified by the probability vector. (b) If the resulting solution is not a weak classifier with the weighted training error, keep taking convex combinations of the solution and the probability vector until the weighted training error drops below the desired threshold. I put a bound on the number of searches to 20 just to ensure that the code doesn't run for too long. 
