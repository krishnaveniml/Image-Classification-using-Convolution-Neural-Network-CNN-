import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------
# Load CIFAR-10 Dataset
# -------------------------

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = y_train.flatten()
y_test = y_test.flatten()

# Validation split

x_val = x_train[:5000]
y_val = y_train[:5000]

x_train = x_train[5000:]
y_train = y_train[5000:]

# -------------------------
# Data Augmentation
# -------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# -------------------------
# CNN Model
# -------------------------

model = models.Sequential([

    layers.Input(shape=(32,32,3)),

    data_augmentation,

    layers.Conv2D(32,3,padding="same",
                  kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.Conv2D(32,3,padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64,3,padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.Conv2D(64,3,padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.MaxPooling2D(),
    layers.Dropout(0.30),

    layers.Conv2D(128,3,padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.MaxPooling2D(),

    layers.Dropout(0.40),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128,activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(10,activation="softmax")

])

# -------------------------
# Compile Model
# -------------------------

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# -------------------------
# Callbacks
# -------------------------

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3
    ),

    ModelCheckpoint(
        "best_model.keras",
        save_best_only=True
    )

]

# -------------------------
# Train Model
# -------------------------

history = model.fit(

    x_train,

    y_train,

    validation_data=(x_val,y_val),

    epochs=30,

    batch_size=128,

    callbacks=callbacks

)

# -------------------------
# Evaluate
# -------------------------

loss,accuracy = model.evaluate(x_test,y_test)

print("\nTest Accuracy:",accuracy)

# -------------------------
# Prediction
# -------------------------

predictions = model.predict(x_test)

y_pred = np.argmax(predictions,axis=1)

print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)

# -------------------------
# Confusion Matrix
# -------------------------

plt.figure(figsize=(8,8))

plt.imshow(cm,cmap="Blues")

plt.title("Confusion Matrix")

plt.colorbar()

plt.xlabel("Predicted")

plt.ylabel("True")

plt.savefig("confusion_matrix.png")

plt.close()

# -------------------------
# Training Curves
# -------------------------

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(history.history["accuracy"],label="Train")

plt.plot(history.history["val_accuracy"],label="Validation")

plt.title("Accuracy")

plt.legend()

plt.subplot(1,2,2)

plt.plot(history.history["loss"],label="Train")

plt.plot(history.history["val_loss"],label="Validation")

plt.title("Loss")

plt.legend()

plt.savefig("training_curves.png")

plt.show()