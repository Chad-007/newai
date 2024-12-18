import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist
import joblib

def train_linear_regression():
    data = pd.read_csv('data/linear_regression_data.csv')
    features = ['Avg. Area Income', 'Avg. Area House Age', 
                'Avg. Area Number of Rooms', 'Area Population']
    X = data[features].values
    y = data['Price'].values  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Linear Regression MSE: {mse}')
    joblib.dump(lr, 'models/linear_regression_model.pkl')
    print('Linear Regression model saved.')
def train_kmeans():
    data = pd.read_csv('data/Iris.csv')
    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)
    joblib.dump(scaler, 'models/kmeans_scaler.pkl')
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    print('K-Means model and scaler saved.')
def train_neural_network(dataset='mnist'):
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'fashion_mnist'.")
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train_cat, epochs=10, validation_data=(x_test, y_test_cat))
    loss, acc = model.evaluate(x_test, y_test_cat)
    print(f'Neural Network Accuracy: {acc}')
    model.save('models/neural_network_model.h5')
    print('Neural Network model saved.')
if __name__ == '__main__':
    print("Training Linear Regression Model...")
    train_linear_regression()
    print("\nTraining K-Means Clustering Model...")
    train_kmeans()
    print("\nTraining Neural Network Model (MNIST)...")
    train_neural_network(dataset='mnist')