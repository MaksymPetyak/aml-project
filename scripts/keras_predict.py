import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Progbar


from datasets import KaggleDR


if __name__ == "__main__":
    import os
    os.sys.path.append('.')

from BCNN import BCNN
from JFnet import JFnet

# --------- Define parameters ---------
preprocessing_function = KaggleDR.standard_normalize
normalization = None

mc_samples = 100
batch_size = 32
# can limit to predict the first n_batches
n_batches = 100
n_classes = 2
last_layer = "layer_17d"

weights_path = "../training_output/bcnn0vs1234.h5"
dataset_dir = "../../output_test" 
labels_path = "../../output_test/testLabels0vs1234.csv"
out_file = "../predict_output/mc_100_kaggledr_0vs1234_bcnn.pkl"

# --------- Load model ---------
model = BCNN(p_conv=0.2, last_layer=last_layer, n_classes=n_classes,
             weights=weights_path
            )

labels = pd.read_csv(labels_path)
labels.image = labels.image.apply(lambda s: s + ".jpeg")
labels.level = labels.level.astype(str)

# data is kept in folders with images with correspdonding csv file with labelss
datagen = ImageDataGenerator(
    preprocessing_function=preprocessing_function,
)

generator = datagen.flow_from_dataframe(
    labels,
    directory=dataset_dir,
    x_col='image',
    y_col='level',
    target_size=(512, 512),
    batch_size=batch_size,
    shuffle=False,
    seed=None,
)

# ---------- Main loop ----------
n_samples = labels.shape[0]
n_out = model.net.output_shape[1]

if n_batches:
    n_samples = min(n_batches * batch_size, n_samples)

det_out = np.zeros((n_samples, n_out), dtype=np.float32)
stoch_out = np.zeros((n_samples, n_out, mc_samples), dtype=np.float32)

idx = 0
n_batch = 0

progbar = Progbar(n_samples)
for X, y in generator:
    if n_batch >= n_batches:
        break

    n_s = X.shape[0]    

    if isinstance(model, JFnet):
        img_dim = (512, 512)
        inputs = [X, img_dim]
    else:
        inputs = [X]

    det_out[idx:idx + n_s] = model.predict(*inputs)
    stoch_out[idx:idx + n_s] = model.mc_samples(*inputs,
                                                T=mc_samples)

    idx += n_s
    n_batch += 1

    progbar.add(n_s)

results = {'det_out': det_out,
           'stoch_out': stoch_out}

with open(out_file, "wb") as out_f:
    pickle.dump(results, out_f)
