# Play-Predictor

This project demonstrates the use of K-Nearest Neighbors (KNN) for predicting whether to play based on weather conditions.

Data Loading-
Use the PlayPredictor.csv file for reading the data.

Data PreProcessing-
•	Dropped Unnamed: 0 to remove irrelevant serial number info.
•	Applied Label Encoding to convert categorical features into numeric format.
•	Used StandardScaler to normalize feature values.

Model building-
•	Split the dataset into training and testing sets using train_test_split (80/20).
•	Trained a KNeighborsClassifier with n_neighbors=3.
•	Predicted outcomes on the test set using the trained model.

Visualization-
•	USed Heatmap from the library Seaborn to check using Corelation.

Evaluation-
•	Measured model accuracy with accuracy_score. 
•	Accuracy score is: 0.83

Suggestions-
Feel free to modify the python code as per your analysis and requirement.




