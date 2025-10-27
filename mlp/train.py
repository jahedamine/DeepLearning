import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint
from tensorflow.keras.regularizers import l2

data = load_digits()
x = data.data
y = to_categorical(data.target , num_classes=10)

scaler = StandardScaler()
x_scalled = scaler.fit_transform(x)

x_train , x_test , y_train , y_test = train_test_split(x_scalled,y , test_size=0.2 , random_state=42)


model = Sequential([
    Dense (128 , activation= "relu" , input_shape=(64,), kernel_regularizer =l2(0.001)),
    Dropout (0.3),
    Dense (64, activation="relu", kernel_regularizer =l2(0.001)),
    Dropout(0.3),
    Dense(10 , activation="softmax")
])
optimizer = Adam(learning_rate=0.01)

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

callback = [
    EarlyStopping( patience=10, restore_best_weights = True),
    ModelCheckpoint ("best_model.keras", save_best_only = True )
]
history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callback,
    verbose=1
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="train acc")
plt.plot(history.history['val_accuracy'], label="val acc")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="train loss")
plt.plot(history.history['val_loss'], label="val loss")
plt.legend()
plt.title("Loss")
plt.show()