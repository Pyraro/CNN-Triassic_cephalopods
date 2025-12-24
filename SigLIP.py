



import os
#os.environ["KERAS_BACKEND"] = "tensroflow" # you can also use tensorflow or torch

import keras_cv
import keras_nlp
import keras
from keras import ops
import tensorflow as tf

import cv2
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt 

import math

from sklearn.model_selection import GroupKFold

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("KerasCV:", keras_cv.__version__)
print("KerasNLP:", keras_nlp.__version__)

#keras.config.set_dtype_policy("mixed_float16")


class CFG:
    debug = False
    seed = 42
    
    image_path = "./Cephalopod_db/publi/Converted_dataset/Image"
    caption_path = "./Cephalopod_db/publi/Converted_dataset"
    
    # Training params
    batch_size = 8
    epochs = 1000

    # Image Encoder
    image_preset = "efficientnetv2_b0_imagenet_classifier"
    image_size = [512, 512]
    
    # Text Encoder
    text_preset = "distil_bert_base_multi"
    sequence_length = 200
    
    # For embedding head
    embedding_dim = 256
    dropout = 0.3

    log_dir='./datasets/logs'

keras.utils.set_random_seed(CFG.seed)

df = pd.read_csv(f"{CFG.caption_path}/captions.csv")
df["image_path"] = CFG.image_path + "/" + df.image
df['pdf_encode'] = pd.factorize(df['id_pdf'])[0] + 1
df.head()



# Create a GroupKFold object with 5 folds
gkf = GroupKFold(n_splits=5)

# Add fold column based on groups
df['fold'] = -1
for fold, (train_index, valid_index) in enumerate(gkf.split(df, groups=df["image"])):
    df.loc[valid_index, 'fold'] = fold

preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
    preset=CFG.text_preset, # Name of the model
    sequence_length=CFG.sequence_length, # Max sequence length, will be padded if shorter
)

def build_augmenter():
    # Define augmentations
    aug_layers = [
        keras_cv.layers.RandomBrightness(factor=0.1, value_range=(0, 1)),
        keras_cv.layers.RandomContrast(factor=0.1, value_range=(0, 1)),
        keras_cv.layers.RandomSaturation(factor=(0.45, 0.55)),
        keras_cv.layers.RandomHue(factor=0.1, value_range=(0, 1)),
        keras_cv.layers.RandomCutout(height_factor=(0.06, 0.15), width_factor=(0.06, 0.15)),
        keras_cv.layers.RandomFlip(mode="horizontal"),
        keras_cv.layers.RandomZoom(height_factor=(0.05, 0.10)),
        keras_cv.layers.RandomRotation(factor=(0.01, 0.05)),
    ]
    
    # Apply augmentations to random samples
    aug_layers = [keras_cv.layers.RandomApply(x, rate=0.5) for x in aug_layers]
    
    # Build augmentation layer
    augmenter = keras_cv.layers.Augmenter(aug_layers)

    # Apply augmentations
    def augment(inp):
        inp["images"] = augmenter({"images": inp["images"]})["images"]
        return inp
    return augment

def build_decoder(target_size=CFG.image_size):
    def decode_image(image_path):
        # Read jpeg image
        file_bytes = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(file_bytes)
        
        # Resize
        image = tf.image.resize(image, size=target_size, method="area")
        
        # Rescale image
        image = tf.cast(image, tf.float32)
        image /= 255.0
        
        # Reshape
        image = tf.reshape(image, [*target_size, 3])
        return image

    def decode_text(text):
        text = preprocessor(text)
        return text
    
    def decode_distance(distance):
        return tf.cast(distance,tf.float32)
    
    def decode_pdf(pdf):
        
        return tf.cast(pdf,tf.int64)

    def decode_input(image_path, text,distance,pdf):
        image = decode_image(image_path)
        text = decode_text(text)
        distance=decode_distance(distance)
        pdf=decode_pdf(pdf)
        return {"images":image, "texts":text,"distance":distance,"pdf":pdf}

    return decode_input


def build_dataset(
    image_paths,
    texts,
    distance,
    pdf,
    batch_size=16,
    cache=True,
    decode_fn=None,
    augment_fn=None,
    augment=False,
    repeat=True,
    shuffle=1024,
    cache_dir="",
    drop_remainder=True,
):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder()

    if augment_fn is None:
        augment_fn = build_augmenter()

    AUTO = tf.data.experimental.AUTOTUNE

    slices = (image_paths, texts, distance,pdf)
    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    ds = ds.cache(cache_dir) if cache else ds
    ds = ds.repeat() if repeat else ds
    if shuffle:
        ds = ds.shuffle(shuffle, seed=CFG.seed)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.prefetch(AUTO)
    return ds

# Sample from full data
sample_df = df.groupby("image").head(1).reset_index(drop=True) # .sample(frac=1.0)
train_df = sample_df[sample_df.fold != 0]
valid_df = sample_df[sample_df.fold == 0]
print(f"# Num Train: {len(train_df)} | Num Valid: {len(valid_df)}")

# Train
train_paths = train_df.image_path.values
train_texts = train_df.caption.values
train_distance = train_df.dist.values
train_pdf=train_df.pdf_encode.values
train_ds = build_dataset(train_paths, train_texts,train_distance,train_pdf,
                         batch_size=CFG.batch_size,
                         repeat=True, shuffle=False, augment=True, cache=False)
#print(train_ds)

# Valid
valid_paths = valid_df.image_path.values
valid_texts = valid_df.caption.values
valid_distance = valid_df.dist.values
valid_pdf=valid_df.pdf_encode.values
valid_ds = build_dataset(valid_paths, valid_texts,valid_distance,valid_pdf,
                         batch_size=CFG.batch_size,
                         repeat=False, shuffle=False, augment=False, cache=False)

batch = next(iter(train_ds))
imgs = batch["images"]
txts = batch["texts"]
dist=batch["distance"]
pdf_file=batch["pdf"]
print(pdf_file)
'''
fig = plt.figure(figsize=(15, 10)) 
for i in range(6):
    img = imgs[i].numpy()
    caption = preprocessor.tokenizer.detokenize(txts["token_ids"][i])#.numpy.decode("utf-8")
    caption = caption.replace("[PAD]","").replace("[CLS]","").replace("[SEP]","").strip()
    caption = " ".join(caption.split(" ")[:12]) + "\n" + " ".join(caption.split(" ")[12:])
    plt.subplot(2, 3, i + 1) 
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption, fontsize=12) 

plt.tight_layout()
plt.show()'''

class SigLIPLoss(keras.losses.Loss):
    def __init__(self, name="siglip_loss"):
        """Calculates the SigLIP loss.

        Standard sigmoid computes the loss twice, once assuming positive
        labels and once assuming negative ones. But in this case, positives
        are on the "me" diagonal and negatives are elsewhere. So, we only
        compute the loss for each once.

        Call Args:
            y_true: Ground truth labels.
            y_pred: Predicted logits.

        Returns:
            tensor: The SigLIP loss.
        """
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Normalize by the number of positives per column (npos), which is one.
        # Since it's one, we just sum.
        loss = -ops.sum(ops.log_sigmoid(y_true * y_pred), axis=-1)

        # NOTE: This is equivalent to concatenating "me" and "ot" along axis -1 above.
        loss = ops.mean(loss)
        return loss

    
class ProjectionHead(keras.Model):
    def __init__(
        self,
        embedding_dim=CFG.embedding_dim,
        dropout=CFG.dropout,
    ):
        super().__init__()
        self.projection = keras.layers.Dense(embedding_dim)
        self.gelu = keras.layers.Activation("gelu")
        self.fc = keras.layers.Dense(embedding_dim)
        self.dropout = keras.layers.Dropout(dropout)
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, x):

        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
def build_image_encoder():
    backbone = keras_cv.models.ImageClassifier.from_preset(
        CFG.image_preset,
    )
    out = backbone.layers[-2].output
    out = ProjectionHead()(out)
    model = keras.models.Model(backbone.input, out)
    return model

image_encoder = build_image_encoder()
image_encoder.summary()

def build_text_encoder():
    #distance_input = keras.Input(shape=(1,), name="distance")
    #print(distance_input)
    backbone = keras_nlp.models.DistilBertClassifier.from_preset(
        CFG.text_preset,
        num_classes=1
    )
    backbone.summary()
    out = backbone.layers[-3].output            
    #out = keras.layers.concatenate([out,distance_input])             #<------ embed distance here ?
    #print(out)
    out = ProjectionHead()(out)
    model = keras.models.Model(backbone.input, out)
    return model

text_encoder = build_text_encoder()
text_encoder.summary()

class SigLIPModelPlus(keras.Model):
    def __init__(
        self,
        image_encoder,
        text_encoder,
        num_logits,
        logit_scale,
        logit_bias,
        distance_weight,
        
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.num_logits = num_logits
        self.logit_scale = logit_scale
        self.logit_bias = logit_bias
        self.distance_weight = distance_weight
        

    def compile(
        self,
        optimizer,
        loss,
    ):
        super().compile(optimizer=optimizer)
        self.loss = loss

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        logits = self.get_logits(x, training=True)
        labels = self.get_ground_truth(self.num_logits)
        return self.loss(labels, logits)

    def get_ground_truth(self, num_logits):
        labels = -ops.ones((num_logits, num_logits))
        labels = labels + 2 * ops.eye(num_logits)
        labels = ops.cast(labels, dtype="float32")
        return labels

    def get_logits(self, x, training):
        image_features = self.image_encoder(x["images"], training=training)
        text_features = self.text_encoder(x["texts"], training=training)
        #image_features = tf.math.l2_normalize(image_features, axis=-1)
        #text_features = tf.math.l2_normalize(text_features, axis=-1)
        logits = image_features @ ops.transpose(text_features)

        dist = tf.reshape(x["distance"], (-1, 1)) 

        dist_norm = tf.math.log(dist)
        dist_norm = dist_norm / tf.reduce_mean(dist_norm)

        alpha = self.distance_weight  
        distance_bias = -alpha * dist_norm 
        logits = logits + distance_bias

        logits = self.logit_scale * logits + self.logit_bias
        logits = ops.cast(logits, dtype="float32")
        return logits

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def call(self, x, training=False):
        return self.get_logits(x, training=training)
    

class SigLIPModel(keras.Model):
    def __init__(
        self,
        image_encoder,
        text_encoder,
        num_logits,
        logit_scale,
        logit_bias,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.num_logits = num_logits
        self.logit_scale = logit_scale
        self.logit_bias = logit_bias

    def compile(
        self,
        optimizer,
        loss,
    ):
        super().compile(optimizer=optimizer)
        self.loss = loss

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        logits = self.get_logits(x, training=True)
        labels = self.get_ground_truth(self.num_logits)
        return self.loss(labels, logits)

    def get_ground_truth(self, num_logits):
        labels = -ops.ones((num_logits, num_logits))
        labels = labels + 2 * ops.eye(num_logits)
        labels = ops.cast(labels, dtype="float32")
        return labels

    def get_logits(self, x, training):
        image_features = self.image_encoder(x["images"], training=training)
        text_features = self.text_encoder(x["texts"], training=training)
        logits = image_features @ ops.transpose(text_features)
        logits = self.logit_scale * logits + self.logit_bias
        logits = ops.cast(logits, dtype="float32")
        return logits

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def call(self, x, training=False):
        return self.get_logits(x, training=training)

#print(image_encoder)
#print(text_encoder)  


def get_lr_callback(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 3e-6, 5e-7 * batch_size, 3e-7
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        elif mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step': lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
        plt.xlabel('epoch'); plt.ylabel('lr')
        plt.title('LR Scheduler')
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create lr callback

class SigLIPAccuracyCallback(keras.callbacks.Callback):
    def __init__(self, train_ds, val_ds, max_batches,accuracy_summary_writer):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.max_batches = max_batches
        self.accuracy_summary_writer=accuracy_summary_writer

    def compute_accuracy(self, dataset):
        correct = 0
        total = 0

        for i, batch in enumerate(dataset):
            if i >= self.max_batches:
                break

            logits = self.model(batch, training=False)
            logits = logits.numpy()

            preds = np.argmax(logits, axis=1)
            labels = np.arange(len(preds))

            correct += np.sum(preds == labels)
            total += len(preds)

        return correct / max(total, 1)

    def on_epoch_end(self, epoch, logs=None):
        train_acc = self.compute_accuracy(self.train_ds)
        val_acc = self.compute_accuracy(self.val_ds)

        logs = logs or {}
        logs["train_acc"] = train_acc
        logs["val_acc"] = val_acc
        
        with self.accuracy_summary_writer.as_default():
            tf.summary.scalar('train_acc', train_acc, step=epoch)
            tf.summary.scalar('val_acc', val_acc, step=epoch)
accuracy_summary_writer = tf.summary.create_file_writer(CFG.log_dir)
acc_callback = SigLIPAccuracyCallback(
    train_ds,
    valid_ds,
    max_batches=len(valid_df),  # limite pour la vitesse
    accuracy_summary_writer=accuracy_summary_writer
)

lr_cb = get_lr_callback(CFG.batch_size, mode="cos",epochs=CFG.epochs, plot=False)

ckpt_cb = keras.callbacks.ModelCheckpoint("./datasets/saved_weights/best_sigLIP_model.weights.h5",
                                         monitor='val_loss',
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='min')

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=CFG.log_dir, update_freq=1, histogram_freq=0,write_graph=False,write_images=False)

model = SigLIPModelPlus(image_encoder, text_encoder, num_logits=CFG.batch_size,
                    logit_scale=2.30, logit_bias=-10.0, distance_weight=0.2)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss=SigLIPLoss())
'''
model = SigLIPModel(image_encoder, text_encoder, num_logits=CFG.batch_size,
                    logit_scale=2.30, logit_bias=-10.0)

model.compile(optimizer="adam", loss=SigLIPLoss())
'''
history = model.fit(
        train_ds,
        epochs=CFG.epochs,
        callbacks=[lr_cb,ckpt_cb,tb_callback,acc_callback],
        steps_per_epoch=len(train_df) // CFG.batch_size,
        validation_data=valid_ds,
        verbose=1,
    )

model.load_weights("./datasets/saved_weights/best_sigLIP_model.weights.h5")

#keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
'''
model.load_weights("./datasets/saved_weights/best_sigLIP_model.weights.h5")

def process_image(path):
    img = cv2.imread(path)[...,::-1] # BGR -> RGB
    img = cv2.resize(img, dsize=CFG.image_size, interpolation=cv2.INTER_AREA)
    img = img / 255.0 
    return img

def process_text(text):
    text = [f"{x}" for x in text]
    return preprocessor(text)

def zero_shot_classifier(image_path, candid_labels):
    image = process_image(image_path)
    plt.imshow(image)
    image = ops.convert_to_tensor(image)[None,]
    text = process_text(candid_labels)
    pred = model({"images":image, "texts":text})
    pred = ops.softmax(pred)*100
    pred = ops.convert_to_numpy(pred).tolist()[0]
    pred = dict(zip(candid_labels, np.round(pred, 2)))
    print(pred)
    plt.title(f"Prediction: {pred}", fontsize=10)
    plt.show()
    return pred

pred = zero_shot_classifier(image_path="./Cephalopod_db/publi/Converted_dataset/Image/page_410.1007@s12542-011-0128-7.jpg",
                            candid_labels=[
                                "Figure 25. Field photograph of part of a ‘cephalopod pocket’ in a loose block of reef limestone of the Vasalemma Formation, Keila Regional Stage, Rummu quarry. Note the chaotic bedding of the fragmented cephalopod shells."
                                , "Table 2. Comparison of successive Early Ordovician nautiloid associations of the Tribes Hill Formation (Skullrockian, NYSM locality 5896) and Rochdale Formation (Stairsian, NYSM locality 5897) at the Smith Basin section, Washington County, New York",
                                  "Figure 1. General morphology of the shell of Lituites. Modified after Furnish & Glenister in Moore (1964, 363, fig. 266-1a).",
                                  "Figure 3. Rummoceras rummuensis gen. et sp. nov., holotype, TUG 1709-31. A and B, general views from ventral and dorsal sides, respectively; C, detail of the siphuncle (lateral section), with thick projections visible at the imner side of the connecting rings; D, trans- verse cross section of the shell and the siphuncle, oriented with apex down."
                                  ])'''