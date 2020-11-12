import tensorflow as tf


fasion = tf.keras.datasets.fashion_mnist

(x_train, y_trian), (x_test, y_test) = fasion.load_data()

x_train, x_test = x_train/255.0, x_test/255.0

# model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_trian, batch_size=32, epochs=15, validation_data=(x_test, y_test), validation_freq=1)

model.summary()