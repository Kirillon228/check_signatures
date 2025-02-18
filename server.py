import uuid
import numpy as np 
import random
import json
import uvicorn
import argparse
import os
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import urllib.request
import io
import binascii
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing_extensions import Annotated
from fastapi import FastAPI, File, Form, UploadFile
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow_addons.losses import metric_learning
from scipy.stats import gaussian_kde
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

import tensorflow as tf
from tensorflow_addons.losses import metric_learning
from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from tensorflow.keras.utils import custom_object_scope
from typeguard import typechecked
from typing import Optional, Union, Callable


def custom_contrastive_loss(
    y_true: TensorLike,
    y_pred: TensorLike,
    margin: FloatTensorLike = 1.0,
    distance_metric: Union[str, Callable] = "L2",
) -> tf.Tensor:
    labels = tf.convert_to_tensor(y_true, name="labels")
    embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

    convert_to_float32 = (
        embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
    )

    # Reshape label tensor to [batch_size, 1].
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    if distance_metric == "L2":
        pdist_matrix = metric_learning.pairwise_distance(
            precise_embeddings, squared=False
        )

    elif distance_metric == "squared-L2":
        pdist_matrix = metric_learning.pairwise_distance(
            precise_embeddings, squared=True
        )

    elif distance_metric == "angular":
        pdist_matrix = metric_learning.angular_distance(precise_embeddings)

    else:
        pdist_matrix = distance_metric(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    losses = tf.square(tf.where(adjacency, pdist_matrix, tf.maximum(margin - pdist_matrix, 0.0)))
    loss = tf.reduce_mean(losses)

    if convert_to_float32:
        return tf.cast(loss, embeddings.dtype)
    else:
        return loss

class CustomContrastiveLoss(LossFunctionWrapper):
    @typechecked
    def __init__(
        self,
        margin: FloatTensorLike = 1.0,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            custom_contrastive_loss,
            name=name,
            reduction=tf.keras.losses.Reduction.NONE,
            margin=margin,
            distance_metric=distance_metric,
        )


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_names = [
  "smallResnet101_dropoutRegularized256_lr0.001_decay0.8_epochs50_lossTripletSemiHardLoss0.5",
  "smallResnet101_regularized256_lr0.0001_decay0.9_epochs50_lossTripletSemiHardLoss0.5",
  "smallEfficientNet_regularized64_lr0.001_decay0.8_epochs50_lossContrastiveLoss_0.5",
  "smallEfficientNet_regularized256_lr0.001_decay0.8_epochs50_lossContrastiveLoss_0.5"
]
models = {}
stats = {}
for i in model_names:
  with custom_object_scope({'CustomContrastiveLoss': CustomContrastiveLoss}):
    models[i] = load_model(f'D:/sigver/{i}/model')
    
  with open(f'D:/sigver/{i}/stats_l2_test.json', 'r') as file:
    stats_two_options = json.load(file)
    stats[i] = { key: gaussian_kde(value) for key, value in stats_two_options.items()}

@app.get("/")
async def index():
    with open("ver.html", encoding="utf-8") as index:
        return HTMLResponse(content=index.read(), status_code=200)
    

@app.post("/check_signatures")
async def check_signatures(
    image1: Annotated[bytes, File()],
    image2: Annotated[bytes, File()],
    option: Annotated[str, Form()],
):            
    loaded_model = models[option]
    model_kde = stats[option]

    stream1 = io.BytesIO(image1)
    stream2 = io.BytesIO(image2)
    img1 = Image.open(stream1)
    img2 = Image.open(stream2)  
      
    img_np1 = prepocessing(img1)
    img_np2 = prepocessing(img2)

    embeddings1 = loaded_model.predict(tf.convert_to_tensor([img_np1]))
    embeddings2 = loaded_model.predict(tf.convert_to_tensor([img_np2]))

    distances_l2 = tf.math.reduce_euclidean_norm(embeddings1 - embeddings2, axis=1)
    label, prob = get_probs(model_kde,distances_l2)
    
    result = CheckResult(predict = label, prob = list(map(lambda x: int(x[0]*100), prob)))
    
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

class CheckResult(BaseModel):
    predict: str 
    prob: list[float]    

priors_two_options = {"positive": 1/2, "negative": 1/2}
def get_two_options_probs(kde, value):
  f_pos = kde['positive'](value)
  f_neg = kde['negative'](value)

  # Compute posterior probabilities using Bayes' theorem
  posterior_pos = f_pos * priors_two_options["positive"]
  posterior_neg = f_neg * priors_two_options["negative"]

  # Normalize to get probabilities
  total = posterior_pos + posterior_neg

  prob_pos = posterior_pos / total
  prob_neg = posterior_neg / total
  probs = [prob_pos, prob_neg]

  # Determine the class with the highest probability
  predicted_class = tf.math.argmax(probs, axis=0)
  label = "Genuine" if predicted_class[0] == 0 else "Non Genuine"

  return label, probs

# Prior probabilities (assume equal if unknown)
priors = {"positive": 1/3, "hard": 1/3, "easy": 1/3}
def get_probs(kde, value):
  f_pos = kde['positive'](value)
  f_hard = kde['hard'](value)
  f_easy = kde['easy'](value)

  # Compute posterior probabilities using Bayes' theorem
  posterior_pos = f_pos * priors["positive"]
  posterior_hard = f_hard * priors["hard"]
  posterior_easy = f_easy * priors["easy"]

  # Normalize to get probabilities
  total = posterior_pos + posterior_hard + posterior_easy

  prob_pos = posterior_pos / total
  prob_hard = posterior_hard / total
  prob_easy = posterior_easy / total
  probs = [prob_pos, prob_hard, prob_easy]

  # Determine the class with the highest probability
  predicted_class = tf.math.argmax(probs, axis=0)

  if predicted_class[0] == 0:
     label = "Оригінал"
  elif predicted_class[0] == 1:
     label = "Підробка"
  else:
     label = "Інше"

  return label, probs

def prepocessing(img: Image):
   img = img.convert('RGB')
   img_res = img.resize((224, 224))
   img_np = np.array(img_res)
   img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
   img_np = threshold_image(img_np)
   img_np = img_np * (1./255)
   return np.dstack([img_np, img_np, img_np])

def threshold_image(img_arr):
  thresh = threshold_otsu(img_arr)
  return np.where(img_arr > thresh, 255, 0) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Alies', description='Alies Game')
    parser.add_argument('ip')
    parser.add_argument('-p', '--port')
    args = parser.parse_args()
    uvicorn.run(app, host=args.ip, port=args.port)