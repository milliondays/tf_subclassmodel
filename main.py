import pathlib
import os,sys
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.applications import EfficientNetB0

def process_path(file_path,CLASS_NAMES,WIDTH=128,HEIGHT=128):
  # label
  parts = tf.strings.split(file_path, os.path.sep)
  label = (parts[-2] == CLASS_NAMES)

  # src
  src = tf.io.read_file(file_path)
  src = tf.image.decode_png(src,channels=3)
  src = tf.image.convert_image_dtype(src, tf.float32)

  # preprocess
  src = tf.image.resize(src, [WIDTH, HEIGHT])
  return src, label




def ds_option(ds, is_train = True, BATCH_SIZE = 4, cache=True, shuffle_buffer_size=1000):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  if is_train:
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  ds = ds.repeat()
  ds = ds.batch(BATCH_SIZE)

  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return ds


def make_ds(data_dir,is_train=True, BATCH_SIZE=4, WIDTH=128, HEIGHT=128):
  """
  :param data_dir: root of Train, Val or Test path
  :param is_train:
  :param BATCH_SIZE:
  :param WIDTH:
  :param HEIGHT:
  :return: dataset
  """
  data_dir = pathlib.Path(data_dir)
  CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
  list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'))
  labeled_ds = list_ds.map(lambda file_path: process_path(file_path,CLASS_NAMES=CLASS_NAMES,WIDTH=WIDTH,HEIGHT=HEIGHT),
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
  final_ds= ds_option(labeled_ds,is_train=is_train,BATCH_SIZE=BATCH_SIZE)
  return final_ds







class YogaPose(tf.keras.Model):
  def __init__(self, num_classes=30, freeze=False):
    super(YogaPose, self).__init__()
    self.base_model = EfficientNetB0(include_top=False, weights='imagenet')
    # Freeze the pretrained weights
    if freeze:
      self.base_model.trainable = False
    self.top = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dropout(0.5, name="top_dropout")])
    self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")

  def call(self, inputs, training=True):
    x = self.base_model(inputs)
    x = self.top(x)
    x = self.classifier(x)
    return x

class Trainer:
  def __init__(self, model, epochs, batch, loss_fn, optimizer):
    self.model = model
    self.epochs = epochs
    self.batch = batch
    self.loss_fn = loss_fn
    self.optimizer=optimizer
  def train(self, train_dataset, train_metric):
    for epoch in range(self.epochs):
      print("\nStart of epoch %d" % (epoch,))
      # Iterate over the batches of the dataset.
      for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
          logits = self.model(x_batch_train, training=True)
          loss_value = self.loss_fn(y_batch_train, logits)
          # tf.print(loss_value)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        # Update training metric.
        train_metric.update_state(y_batch_train, logits)
        # Log every 5 batches.
        if step % 5 == 0:
          print(
            "Training loss (for one batch) at step %d: %.4f"
            % (step, float(loss_value))
          )
          print("Seen so far: %d samples" % ((step + 1) * self.batch))
          print(train_metric.result().numpy())
        # Display metrics at the end of each epoch.
      train_acc = train_metric.result()
      print("Training acc over epoch: %.4f" % (float(train_acc),))


if __name__ == '__main__':
  data_dir = f'/home/veritas/src/train/'
  is_train = True
  BATCH_SIZE = 4
  WIDTH = 128
  HEIGHT = 128
  EPOCH=40

  #dataset
  ds = make_ds(data_dir=data_dir, is_train=is_train, BATCH_SIZE=BATCH_SIZE, WIDTH=WIDTH, HEIGHT=HEIGHT)

  # model load
  model = YogaPose(num_classes=107, freeze=True)
  model.build(input_shape=(None, WIDTH, HEIGHT, 3))




  loss_function = tf.keras.losses.CategoricalCrossentropy()
  optimizer = tf.keras.optimizers.Adam()
  train_acc_metric = tf.keras.metrics.CategoricalAccuracy()

  # trainer
  trainer = Trainer(model=model,
                    epochs=EPOCH,
                    batch=BATCH_SIZE,
                    loss_fn=loss_function,
                    optimizer=optimizer)

  # train
  trainer.train(train_dataset=ds,train_metric=train_acc_metric)
