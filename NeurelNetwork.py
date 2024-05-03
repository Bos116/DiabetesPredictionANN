import pandas as pd
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from keras.optimizers import Adam

#Loading dataset
dataset = pd.read_csv('DatasetOfDiabetes.csv')

#formating columns
dataset['CLASS'] = dataset['CLASS'].replace({'N ':'N','Y ': 'Y'})
dataset['Gender'] = dataset['Gender'].replace({'f': 'F'})

# Here i split the data into two variables X and Y
X = pd.DataFrame(dataset.iloc[:, 2:13].values)
Y = dataset.iloc[:, 13].values

# Here i encode catagorical data
label_encoder_X_0 = LabelEncoder()
X.loc[:, 0] = label_encoder_X_0.fit_transform(X.iloc[:, 0])

# here i one hot encode the Y data
encoder = OneHotEncoder(categories='auto', sparse_output=False)
Y = encoder.fit_transform(Y.reshape(-1, 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 

#here i perform feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#here i define my layers in my neurel network
classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=3, kernel_initializer='uniform', activation='softmax'))

# Define the learning rate
learning_rate = 0.01  # Example learning rate, you can adjust it as needed

# Compile the model with the specified learning rate
opt = Adam(learning_rate=learning_rate)
classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


#fit my model
classifier.fit(X_train, Y_train, batch_size=10, epochs=60, verbose=0)

#predict the test and print results
Y_pred= classifier.predict(X_test)
Y_pred=(Y_pred > 0.5)
cm = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test.argmax(axis=1), Y_pred.argmax(axis=1), average='weighted')

print("Confusion Matrix:\n*",cm)
print("Accuracy:", accuracy)
print("precision:", precision)