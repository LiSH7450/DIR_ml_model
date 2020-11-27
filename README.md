# DIR_ml_model
This repository is to provide the dataset and code to the article 'A QSAR Model for the Severity Prediction of Drug-induced Rhabdomyolysis by using Random Forest'. Please note that this code is ONLY provided for academic use.

file list:

    1.model_Most_40_new.py : The code for model building with four methods.
  
    2.freq_analysis.py : The code for method comparison.
  
    3.Model_FAERS_RF.py : The code for final RF model and external verification.
  
folder list:

    1.Data : The data  we used in model building.
  
    2.model : The final RF , RF_permu_X, RF_permu_Y  model.
  
Note：

    1.The code for model_Most_40_new.py yield some result files neventually. These files need to be further analyzed using freq_analysis.py .
  
    2.Data ,code and models need to be placed under the same path.
  
Python model requirement:

    1.xgboost
    
    2.sklearn
    
    3.pandas
    
    4.numpy
    
    5.joblib
    
    6.seaborn
    
    7.matplotlib
