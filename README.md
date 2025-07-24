#Play-Predictor

This project demonstrates the use of the K-Nearest Neighbors (KNN) algorithm to predict whether to play a game based on weather conditions.

#Data Loading

* Utilized the `PlayPredictor.csv` dataset.
* Loaded using `pandas` for easy manipulation and analysis.


#Data Preprocessing

* Dropped the `Unnamed: 0` column to remove irrelevant serial number information.
* Applied **Label Encoding** to convert categorical features (like Weather, Temperature) into numeric format.
* Normalized feature values using **StandardScaler** to ensure equal weightage across features.

#Model Building

* Split the dataset into training and testing sets using an **80/20** ratio via `train_test_split`.
* Trained a **KNeighborsClassifier** with `n_neighbors=3`.
* Predicted outcomes on the test set using the trained model.


#Visualization

* Generated a **heatmap** using the Seaborn library to visualize feature correlations.

#Evaluation

* Measured model performance using accuracy_score.
* Accuracy-(83%)

#Suggestions

* Feel free to enhance the model by:

  * Trying different classification algorithms (e.g., Decision Trees, SVM).
  * Tuning hyperparameters of KNN.
  * Adding or engineering new features.
  * Using cross-validation for better evaluation.
* Modify the Python code based on your own analysis and domain understanding.

