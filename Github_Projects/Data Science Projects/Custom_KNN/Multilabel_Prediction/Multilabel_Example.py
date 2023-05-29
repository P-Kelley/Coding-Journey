from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from Minimum_Vote_Multilabel_KNN import KNNClassifier

iris = pd.read_csv(r'C:\Users\Mike\Self_Made_Code\AI\Kaggle Competition\Bioinformatics\Custom_KNN\iris.csv')

iris['Name_Code'] = pd.Categorical(iris['Name']).codes

#Extra data generation assigning semi random labels based off index. The classifer will not be accruate in predicting these, they serve as an example
iris['Extra data'] = iris.index % 5
iris['Extra data extended'] = iris.index % 3



X = iris.iloc[:,:4]
y = iris.iloc[:,5:]
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)


mymodel = KNNClassifier(metric = euclidean_distances, n_neighbors=5, minvote = 0)
mymodel.fit(X_train, y_train)
prediction = mymodel.score(X_test, y_test)


print("Accurcy of predicting flower names: {}. Accuracy of predicting semi random data 1: {}. Accuracy of predicting semi random data 2: {}".format(prediction[0], prediction[1], prediction[2]))
