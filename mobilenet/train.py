import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models, callbacks, layers, applications
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tensorflow.keras.models import Model

# Chargement des données CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalisation
x_train, x_test = x_train / 255.0, x_test / 255.0

# Redimensionnement à 96x96
x_train = tf.image.resize(x_train, [96, 96])
x_test = tf.image.resize(x_test, [96, 96])

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Chargement de MobileNetV2 (sans la tête)
base_model = applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Modèle complet avec tête personnalisée
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.25),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Callbacks
earlystop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5)

# Entraînement initial (avec base gelée)
history = model.fit(x_train, y_train,
                    batch_size=32,
                    validation_data=(x_test, y_test),
                    callbacks=[earlystop, reduce_lr],
                    epochs=10,
                    verbose=0)

# Fine-tuning : dégel partiel du backbone
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.006),
              metrics=["accuracy"])

fine_tune_history = model.fit(x_train, y_train,
                              batch_size=32,
                              validation_data=(x_test, y_test),
                              callbacks=[earlystop, reduce_lr],
                              epochs=10,
                              verbose=0)

# Évaluation finale
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n Accuracy après fine-tuning : {acc:.4f}")

# Affichage des courbes d'apprentissage
train_acc = history.history.get("accuracy", []) + fine_tune_history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", []) + fine_tune_history.history.get("val_accuracy", [])

plt.plot(train_acc, label="Train Acc")
plt.plot(val_acc, label="Val Acc")
plt.title("Évolution de la précision")
plt.xlabel("Époque")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()

# Prédictions
y_pred_probs = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = y_test.reshape(-1)

# Matrice de confusion
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de Confusion – MobileNetV2 Fine-tuné")
plt.show()

# Rapport de classification
print(" Rapport de classification :\n")
print(classification_report(y_true, y_pred_classes, target_names=class_names))
print("F1-score (macro) :", f1_score(y_true, y_pred_classes, average='macro'))

#  Visualisation des filtres de la première couche convolutive
layer_outputs = [layer.output for layer in base_model.layers if isinstance(layer, layers.Conv2D)]
activation_model = Model(inputs=base_model.input, outputs=layer_outputs)
img = x_test[1]
input_img = np.expand_dims(img, axis=0)
activations = activation_model.predict(input_img)

first_layer_activation = activations[0]
fig, axes = plt.subplots(3, 6, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < first_layer_activation.shape[-1]:
        ax.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
        ax.axis('off')
plt.suptitle("🔬 Filtres Convolutifs – 1ère couche MobileNetV2")
plt.tight_layout()
plt.show()

#  Fonction Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap + tf.keras.backend.epsilon())
    return heatmap.numpy()

# Grad-CAM :
img = x_test[5]
img_array = np.expand_dims(img, axis=0)
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1")

# Superposition
img_uint8 = np.uint8(255 * img)
heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
superimposed_img = heatmap_color * 0.4 + img_uint8

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Image Originale")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap_resized, cmap='jet')
plt.title("Carte Grad-CAM")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(np.uint8(superimposed_img))
plt.title("Superposition Heatmap")
plt.axis("off")
plt.tight_layout()
plt.show()

# 💾 Sauvegarde finale
model.save("mobilenetv2_cifar10.keras")
print(" Modèle sauvegardé avec succès.")