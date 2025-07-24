import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler

def PlayPredictor(Datapath):
    #load the data
    df=pd.read_csv(Datapath)

    #Analyse the data
    #Analysis-
    #1.Check for null values
    print("Null values:\n", df.isnull().sum())
    #print the values to check numeric/categorical data
    print("Before dropping:")
    print(df.head())
    
    #2.Need to drop the sr no column=Unnamed: 0
    df=df.drop(columns=['Unnamed: 0'])
    print("After dropping Unnamed cloumn:")
    print(df.head())

    #label encoding for transforming values to numeric 
    le= LabelEncoder()
    df["Whether"]=le.fit_transform(df["Whether"])
    df["Temperature"]=le.fit_transform(df["Temperature"])
    df["Play"]=le.fit_transform(df["Play"])

    print("Data after encoding:")
    print(df.head())
    
    #Split the data
    x= df.drop("Play",axis=1)#input variables take everything except x
    y=df["Play"]#target variable 

    #Standard scaler
    scaler=StandardScaler()
    x_scale=scaler.fit_transform(x)

    x_train,x_test,y_train,y_test=train_test_split(x_scale,y,test_size=0.2,random_state=42)
    model=KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train,y_train)#pass training datasets
    y_pred=model.predict(x_test)#testing one

    #accuracy score 
    accuracy=accuracy_score(y_test,y_pred)
    print("Accuracy score is:",accuracy)

    #Data visualisation
    plt.figure(figsize=(6,4))
    sns.heatmap(df.corr(), annot=True, cmap='viridis')
    plt.title("Feature Correlation Heatmap")
    plt.show()
    



def main():
    PlayPredictor("PlayPredictor.csv")

if __name__=="__main__":
    main()
