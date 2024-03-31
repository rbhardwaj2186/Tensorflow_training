import tensorflow as tf
import os
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers, Model
print(tf.__version__)

# Helper Libraries
import numpy as np
import matplotlib.pyplot as plt
#iris = fetch_ucirepo(id=53)
#print(iris)

df = pd.read_csv('D:\Work\Gre\\UTD\Courses\Spring_II\Exams\Tensorflow_developer\Python_3.9\\tfexam\\tfExamTest6\iris.csv')

df_features = df.copy()
df_labels = df_features.pop('variety')

# BUILD PREPROCESSING MODEL
inputs = {}

for name, column in df_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

print(inputs)

# PREPROCESSING NUMERIC INPUTS
numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(df[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)


preprocessed_inputs = [all_numeric_inputs]

# PREPROCESSING CATEGORICAL INPUTS
for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = layers.StringLookup(vocabulary=np.unique(df_features[name]))
  one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
df_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
#tf.keras.utils.plot_model(model = df_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

df_features_dict = {name: np.array(value)
                         for name, value in df_features.items()}

features_dict = {name:values[:1] for name, values in df_features_dict.items()}
df_preprocessing(features_dict)

def df_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(3)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam())
  return model


df_model = df_model(df_preprocessing, inputs)
df_model.fit(x=df_features_dict, y=df_labels, epochs=10)



titanic_model = titanic_model(titanic_preprocessing, inputs)

