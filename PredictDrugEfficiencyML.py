import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import accumulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#################################################################################
# Test on miRNA data
def miRNAData():
    # Read data
    drugData = pd.read_csv("./Inputs/DrugData.csv")
    miRNA = pd.read_csv("./Inputs/miRNA.csv")

    # Use dot instead of coma in every column except the first
    miRNA = miRNA.apply(lambda x: x.str.replace(',','.'))
    # Convert data from string to float in every column except the first two
    miRNA.iloc[:,2:] = miRNA.iloc[:,2:].astype(float).fillna(0.0)
    # Quantate the data into 3 values from 0 to 2
    drugData['IC50 (Quant)'] = pd.cut(x=drugData['IC50 (uM)'], bins=3, labels=False, duplicates='drop')

    drugs = drugData['Compound'].unique()

    for drug in drugs:
        print("Drug Name:", drug)
        preProcess(drug, drugData, miRNA)
        pass

    pass

# Creates a dataset which shows the miRNA values with IC50 for each Cell Lines
def preProcess(drugName, drugData, miRNA):
    #-------------------------------------- Work with Drug data --------------------------------------
    drugData = drugData[['CCLE Cell Line Name','Compound', 'IC50 (uM)', 'IC50 (Quant)']]
    drugData = drugData.loc[drugData['Compound'] == drugName]

    #-------------------------------------- Work with miRNA data --------------------------------------
    # Leaves the Description as index, but i have to live with it..
    miRNA = miRNA.set_index('Description').T
    # this is where the IC50 values for a specific med will go
    miRNA[drugName] = np.nan

    # ---------------------Fill the new column with the IC50 values of the selected drug.---------------
    # Get cell lines of drug data
    cellLines = drugData['CCLE Cell Line Name'].values.tolist()
    for cellLine in cellLines:
        # Get IC50 value for this cell line
        ic50Val = drugData[drugData['CCLE Cell Line Name'] == cellLine]['IC50 (Quant)']
        # Set IC50 value 
        miRNA.loc[miRNA.index == cellLine, drugName] = float (ic50Val)
        pass

    # Remove rows where data is missing
    miRNA = miRNA[miRNA[drugName].notnull()]
    #Get the features names
    features = list(miRNA.iloc[:,:-1])

    # Apply PCA
    principalDf = applyPCA(miRNA, features, drugName)
    # Apply some ML
    applyML(principalDf, miRNA, drugName)

    pass



#Testing copyNumberData
def copyNumberData():
    copyNumbData = pd.DataFrame()
    DrugData = pd.DataFrame()
    copyNumbData = pd.read_csv("./Inputs/CopynumberByGene.csv")
    drugData = pd.read_csv("./Inputs/DrugData.csv")

    # Quantate the IC50 values
    drugData['IC50'] = pd.cut(x=drugData['IC50 (uM)'], bins=3, labels=False, duplicates='drop')
    
    # Transpose data so the features will be columns, and remove unecessary columns
    copyNumbData.set_index('SYMBOL', inplace=True)
    copyNumbData = copyNumbData.T
    copyNumbData = copyNumbData.drop(['EGID', 'CHR', 'CHRLOC', 'CHRLOCEND'])

    # Covert , to . and string data to float
    copyNumbData = copyNumbData.apply(lambda x: x.str.replace(',','.'))
    copyNumbData = copyNumbData.astype(float).fillna(0.0)

    # Save the features for this test.
    features = list(copyNumbData)

    # Add Ic50 values for every drug to the data frame
    drugs = drugData['Compound'].unique()
    for drug in drugs:
        copyNumbData[drug] = np.nan
        
        cellLines = drugData.loc[drugData['Compound'] == drug, 'CCLE Cell Line Name'].values.tolist()
        for cellLine in cellLines:
            ic50Val = drugData.loc[  (drugData['CCLE Cell Line Name'] == cellLine) 
                                   & (drugData['Compound'] == drug), 'IC50'].values
            
            if(len(ic50Val) == 0):
                ic50Val = np.nan
            
            copyNumbData.loc[copyNumbData.index == cellLine, drug] = float (ic50Val)
            pass

        # Create filtered data frames, with features and one drug IC50 result
        filter_col = list(features)
        filter_col.append(drug)
        # Remove cell lines where we have no IC50 value
        filtered = copyNumbData[filter_col]
        filtered = filtered.loc[filtered[drug].notnull()]

        #Run PCA
        principalDf = applyPCA(filtered, features, drug)
        applyML(principalDf,filtered,drug)
        pass

    print(copyNumbData.head(10))
    print(drugData.head(10))

    pass



#Connection the copyNumberData with miRNA data to create and test a dataset
def connectedData():
    # Read input files
    copyNumbData = pd.read_csv("./Inputs/CopynumberByGene.csv")
    drugData = pd.read_csv("./Inputs/DrugData.csv")
    miRNA = pd.read_csv("./Inputs/miRNA.csv")

    # Process the input data so it will be easier to use later.
    # We could do this steps here once, even if we are not sure there are no
    # Data Frames yet, because otherwise it would do it for every different 
    # Data Frame and that would be very slow.
    print("Pre Processing..")

    # Transpose data so the features will be columns, and remove unecessary columns
    copyNumbData.set_index('SYMBOL', inplace=True)
    copyNumbData = copyNumbData.T
    copyNumbData = copyNumbData.drop(['EGID', 'CHR', 'CHRLOC', 'CHRLOCEND'])

    # Transpose data so the features will be columns, and remove unecessary columns
    miRNA.set_index('Description', inplace=True)
    miRNA = miRNA.T

    # Covert , to . and string data to float
    copyNumbData = copyNumbData.apply(lambda x: x.str.replace(',','.'))
    copyNumbData = copyNumbData.astype(float).fillna(0.0)

    # Covert , to . and string data to float
    miRNA = miRNA.apply(lambda x: x.str.replace(',','.'))
    miRNA = miRNA.iloc[:,2:] = miRNA.iloc[:,2:].astype(float).fillna(0.0)

    # For every drug do the test
    drugs = drugData['Compound'].unique()
    for drug in drugs:
        print("\nWorking with:", drug)
        # File name and location to save and read Data Frames
        dataFrameName = 'connected_'+ str(drug) +'.csv'
        dataFrameLocation = './DataFrames/'

        # Quantate the IC50 values
        drugData[drug] = pd.cut(x=drugData['IC50 (uM)'], bins=3, labels=False, duplicates='drop')

        # Try to read in the Data Frame it not successful that means it is not ready
        # so it has to make it.
        try:
            # Check if the selected features data is already ready or not
            new_DF = pd.read_pickle(dataFrameLocation + dataFrameName)
            pass

        except FileNotFoundError:
            print("Creating Dataframe: ", dataFrameName)
            #Creates the connected dataset
            
            # Get cell lines which common in all of 3 files
            drugData_CL = drugData.loc[drugData['Compound'] == drug, 'CCLE Cell Line Name'].values.tolist()
            copyNumbData_CL = copyNumbData.index.tolist()
            miRNA_CL = miRNA.index.tolist()
            common_CL = list(set(drugData_CL) & set(copyNumbData_CL) & set(miRNA_CL))

            #Sorting by index value
            miRNA.sort_index(inplace=True)
            copyNumbData.sort_index(inplace=True)

            #Create new dataframes for storing drug specific data
            miRNA_DF = pd.DataFrame()
            miRNA_DF = miRNA.loc[miRNA.index.isin(common_CL)]
            copyNumbData_DF = pd.DataFrame()
            copyNumbData_DF = copyNumbData.loc[copyNumbData.index.isin(common_CL)]
            drugData_DF = pd.DataFrame()
            drugData_DF = drugData.loc[  (drugData['CCLE Cell Line Name'].isin(common_CL)) 
                                        & (drugData['Compound'] == drug)]

            drugData_DF = drugData_DF.loc[:,['CCLE Cell Line Name', drug]]
            drugData_DF.set_index('CCLE Cell Line Name', inplace=True)
            drugData_DF.sort_index(inplace=True)

            #Concatenate all of this new dataframes into one
            new_DF = pd.concat([miRNA_DF, copyNumbData_DF, drugData_DF], axis=1)
            
            # Save dataFrame
            new_DF.to_pickle(dataFrameLocation + dataFrameName)

            pass

        #Get the features names
        features = list(new_DF.iloc[:,:-1])
        # Apply PCA
        principalDf = applyPCA(new_DF, features, drug)
        #Apply ML
        applyML(principalDf, new_DF, drug)

    pass

# Apply PNC to find the most important 80 components
def applyPCA(dataFrame, features, output):
    print("Input Data:", dataFrame.shape)
    #Separate input and output data
    x = dataFrame.loc[:, features].values
    y = dataFrame.loc[:, [output]].values

    # Standardise data
    x = StandardScaler().fit_transform(x)

    # Have 80 features
    pca = PCA(n_components=80)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents)

    print("Explained Variance r:", sum(pca.explained_variance_ratio_))

    # Plot Variance ration
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    #plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig('Figures/ExplainedVar_'+str(output)+'.png')
    plt.close()

    return principalDf

# Apply KNeighborsClassifier classifier
def applyML(principal, dataFrame, outputCol):

    # Reset index to be able to copy column
    dataFrame = dataFrame.reset_index(drop=True)

    principal['Y'] = dataFrame[outputCol]

    train, test = train_test_split(principal, test_size=0.2)

    #####################
    # Nearest Neighbors #
    #####################

    neigh = KNeighborsClassifier(n_neighbors = 3)
    neigh.fit(train.iloc[:,:-1], train['Y'])
    predicted = neigh.predict(test.iloc[:,:-1])
    score_neigh = neigh.score(test.iloc[:,:-1], test['Y'])
    print("KN Score:", score_neigh)

    # Confusion Matrix
    cnf_matrix = confusion_matrix(test['Y'], predicted)
    np.set_printoptions(precision=2)
    # Plotting it
    plt.figure()
    class_names = ['High response','Normal response','Low response']
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
    plt.savefig('Figures/CM/ConfusionMatrix_'+str(outputCol)+'.png')
    plt.close()

    pass

# Donwloaded from sckikit learn
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Entry point
if __name__ == "__main__":

    #miRNAData()
    #copyNumberData()
    connectedData()

    pass