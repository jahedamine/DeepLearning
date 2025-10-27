import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix ,classification_report
from tensorflow.keras import datasets , models , layers , callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import os

(x_train , y_train),(x_test , y_test)=datasets.cifar10.load_data()

x_train , x_test = x_train/255.0 , x_test/255.0

# Affichage de quelques images pour vérification
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(10, activation="softmax")
])
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


earlystop  = callbacks.EarlyStopping(monitor="val_loss",patience=5, verbose=1 , restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3,factor = 0.2, max_lr=0.0001)

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
)
datagen.fit(x_train)

log_dir = os.path.join("logs", "cnn_aug_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    validation_data=(x_test, y_test),
                    epochs=15,
                    callbacks=[earlystop, reduce_lr, tensorboard_cb])

# Évaluation sur test
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("\n✅ Test Accuracy :", test_acc)

# Matrice de confusion + classification report
y_pred_probs = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = y_test.reshape(-1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Prédit"); plt.ylabel("Réel")
plt.title("Matrice de Confusion")
plt.show()

print("\n📊 Rapport de Classification :")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# 8. Courbes d’apprentissage
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.title("Évolution de la Précision")
plt.xlabel("Époques")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

#affichage des filtre 

img = x_test[0]
input_img = np.expand_dims(img, axis=0)

layer_outputs = [layer.output for layer in model.layers if isinstance(layer, layers.Conv2D)]
model_activation = Model(inputs = model.inputs , outputs = layer_outputs)

activation=model_activation.predict(input_img)

first_layer_activation = activation[0]
fig, axes = plt.subplots(3, 6, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < first_layer_activation.shape[-1]:
        ax.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
        ax.axis('off')
plt.suptitle("Activations de la 1ère couche (filtres convolutifs)")
plt.tight_layout()
plt.show()