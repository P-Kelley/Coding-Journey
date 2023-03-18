from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import pandas as pd

#To test this model just execute the python code and it should display training start/stop and accuracy ratings
#There is also a single prediction section that can take data about a penguin and use the trained classifier to make a best guess, by default this is commented out


df = sns.load_dataset('penguins')

df.dropna(inplace = True)

df['sex_int'] = df['sex'].map({'Male':0, 'Female':1})

one_hot = OneHotEncoder()
encoded = one_hot.fit_transform(df[['island']])
df[one_hot.categories_[0]] = encoded.toarray()

df.drop(['island', 'sex'], axis = 1 , inplace = True)


X = df.iloc[:, 1:]
encoded = one_hot.fit_transform(df[['species']])
df[one_hot.categories_[0]] = encoded.toarray()
y = df[one_hot.categories_[0]]



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,  random_state=100)

model = RandomForestRegressor(n_estimators = 100, random_state=100)


print('Starting training for RF')
model.fit(X_train, y_train)
print('Training completed for RF')

yhat = model.predict(X_test)
print('RF scores - R2:{} MAE:{}'.format( r2_score(y_test,yhat), mean_absolute_error(y_test, yhat)))


'''
single_predict = {'bill_length_mm': 36, 'bill_depth_mm': 17, 'flipper_length_mm':190, 'body_mass_g':3899, 'sex_int':0, 'Biscoe':0, 'Dream':0, 'Torgersen':1}
single_predict = pd.DataFrame(single_predict, index = [0])

single_guess = model.predict(single_predict)

single_guess = pd.DataFrame(single_guess, columns = ['Adelie', 'Chinstrap', 'Gentoo'])

guess_ans = single_guess.drop(columns = single_guess.columns[(single_guess != 1.0).any()])

print('The classifier thinks this is a {} penguin'.format(guess_ans.columns[0]))
'''