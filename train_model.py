import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (150,150)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

model = tf.keras.models.load_model("model/deep_high_accuracy.h5")

model.fit(train_data, validation_data=val_data, epochs=3)

model.save("model/deep_high_accuracy.h5")