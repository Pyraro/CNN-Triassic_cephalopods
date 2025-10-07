""" Title: Denoising Diffusion Implicit Models
Author: [András Béres](https://www.linkedin.com/in/andras-beres-789190210)
Date created: 2022/06/24
Last modified: 2022/06/24
Description: Generating images of flowers with denoising diffusion implicit models.
Accelerator: GPU
"""

""" ## Introduction
...
(kept same header text as original)
"""
# --- SETUP ---
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers
from keras import ops
import numpy as np

# --- HYPERPARAMETERS ---
# data
dataset_repetitions = 10
num_epochs = 1000  # train for at least 50 epochs for good results
image_size = 128

# KID = Kernel Inception Distance, see related section
kid_image_size = 135
kid_diffusion_steps = 5
plot_diffusion_steps = 20

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2

# class-conditioning: minimal addition
num_classes = 10  # change to number of classes you want to condition on

# optimization
batch_size = 32
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4

""" ## Data pipeline
This example previously used only images. We now yield (image, label),
where label is derived deterministically from the file path (hashed bucket).
"""

# load dataset — adapt this path to your dataset
dataset_dir = "./Triassic_Cephalopod/Dataset_croped/Dataset_lateral/train/"

# Create datasets with a 15% validation split
train_dataset, val_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.15,
    subset='both',  # we will split into train/val below
    seed=123,
    image_size=(image_size, image_size),
    batch_size=batch_size,
    label_mode="int",  # return integer labels
    shuffle=True,
)

class_names = train_dataset.class_names
num_classes = len(class_names)

for images, labels in train_dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    for i in range(min(9, images.shape[0])):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(train_dataset.class_names[labels[i]])
        plt.axis("off")
    plt.savefig("./checkpoints/img_ex.jpg")

normalization_layer = keras.layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))



# For performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.repeat(dataset_repetitions).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)



# Visualize examples


@keras.saving.register_keras_serializable()
class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")
        # a pretrained InceptionV3 is used without its classification layer
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = ops.cast(ops.shape(features_1)[1], dtype="float32")
        return (features_1 @ ops.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)
        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(generated_features, generated_features)
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]  # dynamic batch size
        batch_size_f = tf.cast(batch_size, dtype="float32")

        mean_kernel_real = ops.sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = ops.sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = ops.mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()


# --- Network architecture (minimal change: add class input + embedding) ---
@keras.saving.register_keras_serializable()
def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = ops.exp(
        ops.linspace(
            ops.log(embedding_min_frequency),
            ops.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = ops.cast(2.0 * math.pi * frequencies, "float32")
    embeddings = ops.concatenate(
        [ops.sin(angular_speeds * x), ops.cos(angular_speeds * x)], axis=3
    )
    return embeddings


class ClassConditionedSEResidualBlock(keras.layers.Layer):
    def __init__(self, width, num_classes, se_ratio=0.25, **kwargs):
        super().__init__(**kwargs)  # ✅ accept keras serialization args
        self.width = width
        self.num_classes = num_classes
        self.se_ratio = se_ratio


        # --- Standard Residual Components ---
        self.norm1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(width, 3, padding="same")
        self.norm2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(width, 3, padding="same")

        # Skip projection for matching dimensions
        self.skip = layers.Conv2D(width, 1, padding="same")

        # --- Squeeze-and-Excitation ---
        se_channels = max(1, int(width * se_ratio))
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(se_channels, activation="relu")
        self.fc2 = layers.Dense(width, activation=None)  # no activation yet
        self.reshape = layers.Reshape((1, 1, width))

        # --- Class conditioning ---
        self.class_embedding = layers.Embedding(num_classes, width)
        self.class_dense = layers.Dense(width, activation=None)

    def call(self, x, class_labels=None, training=False):
        """
        Args:
            x: Input feature map [B, H, W, C]
            class_labels: Tensor of shape [B] (class indices)
        """
        residual = self.skip(x)

        # --- Residual path ---
        x = self.norm1(x, training=training)
        x = tf.nn.swish(x)
        x = self.conv1(x)

        x = self.norm2(x, training=training)
        x = tf.nn.swish(x)
        x = self.conv2(x)

        # --- SE path ---
        se = self.global_pool(x)  # [B, C]
        se = self.fc1(se)         # [B, se_channels]
        se = self.fc2(se)         # [B, C]

        # --- Inject class conditioning into SE ---
        if class_labels is not None:
            class_emb = self.class_embedding(class_labels)   # [B, C]
            class_emb = self.class_dense(class_emb)          # [B, C]
            se = se + class_emb                              # combine before sigmoid

        se = tf.nn.sigmoid(se)
        se = self.reshape(se)      # [B, 1, 1, C]
        x = x * se                 # channel-wise scaling

        return x + residual
    def get_config(self):
        config = super().get_config()
        config.update({
            "width": self.width,
            "num_classes": self.num_classes,
            "se_ratio": self.se_ratio,
        })
        return config




def DownBlock(width, block_depth,class_labels):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ClassConditionedSEResidualBlock(width, num_classes)(x, class_labels)
        skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth,class_labels):
    def apply(x):
        x, skips = x
        skip = skips.pop()
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        # Now both x and skip should have the same resolution
        x = layers.Concatenate()([x, skip])
        for _ in range(block_depth):
            x = ClassConditionedSEResidualBlock(width, num_classes)(x, class_labels)
        return x
    return apply



def get_network(image_size, widths, block_depth, num_classes):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))
    class_labels = keras.Input(shape=(), dtype="int32")  # scalar per image

    # sinusoidal embedding of noise variance
    e = layers.Lambda(sinusoidal_embedding, output_shape=(1, 1, embedding_dims))(
        noise_variances
    )
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    # class embedding -> expand to spatial map and upsample to image_size
    class_emb = layers.Embedding(input_dim=num_classes, output_dim=embedding_dims)(
        class_labels
    )  # (batch, embedding_dims)
    class_emb = layers.Reshape((1, 1, embedding_dims))(class_emb)
    class_emb = layers.UpSampling2D(size=image_size, interpolation="nearest")(class_emb)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    # concatenate x with time embedding and class embedding
    x = layers.Concatenate()([x, e, class_emb])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth,class_labels)([x, skips])
    for _ in range(block_depth):
        x = ClassConditionedSEResidualBlock(width, num_classes)(x, class_labels)
    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth,class_labels)([x, skips])
    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances, class_labels], x, name="residual_unet")


# --- Diffusion model (changed to accept class labels and pass them through) ---
@keras.saving.register_keras_serializable()
class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth, num_classes):
        super().__init__()
        self.normalizer = layers.Normalization()
        self.image_size = image_size
        self.network = get_network(image_size, widths, block_depth, num_classes)
        self.ema_network = keras.models.clone_model(self.network)
        self.num_classes = num_classes

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        return ops.clip(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = ops.cast(ops.arccos(max_signal_rate), "float32")
        end_angle = ops.cast(ops.arccos(min_signal_rate), "float32")
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        # angles -> signal and noise rates
        signal_rates = ops.cos(diffusion_angles)
        noise_rates = ops.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1
        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training, class_labels):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network
        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates ** 2, class_labels], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps, class_labels):
        # reverse diffusion = sampling

        step_size = 1.0 / float(diffusion_steps)  # convert to float
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # at the first sampling step, the "noisy image" is pure noise
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images
            diffusion_times = ops.ones((tf.shape(noisy_images)[0], 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False, class_labels=class_labels
            )   

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)

        return pred_images

    def generate(self, num_images=None, diffusion_steps=20, class_labels=None):
        if not isinstance(diffusion_steps, int):
            raise ValueError(f"diffusion_steps must be int, got {type(diffusion_steps)}")
        # Infer batch size
        if class_labels is not None:
            batch_size = tf.shape(class_labels)[0]
        elif num_images is not None:
            batch_size = num_images
            class_labels = tf.zeros((batch_size,), dtype=tf.int32)  # default to class 0
        else:
            raise ValueError("Must provide either num_images or class_labels")

        initial_noise = keras.random.normal(
            shape=(batch_size, self.image_size, self.image_size, 3)
        )
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps, class_labels)
        return self.denormalize(generated_images)




    def train_step(self, data):
        # data is (images, labels)
        images, labels = data
        # infer dynamic batch size from actual batch
        batch = tf.shape(images)[0]

        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = keras.random.normal(shape=(batch, self.image_size, self.image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True, class_labels=labels
            )
            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}


    def test_step(self, data):
        # data is (images, labels)
        images, labels = data
        # dynamic batch size
        batch = tf.shape(images)[0]

        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = keras.random.normal(shape=(batch, self.image_size, self.image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False, class_labels=labels
        )
        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        images_denorm = self.denormalize(images)
        # generate images conditioned on the same labels
        generated_images = self.generate(
        diffusion_steps=kid_diffusion_steps, class_labels=labels
        )
        self.kid.update_state(images_denorm, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, save_path="./checkpoints/generated_grid.png", diffusion_steps=20):
        """
        Generate one image per class and plot them in a grid.
        """
        if not isinstance(diffusion_steps, int):
            raise ValueError(f"diffusion_steps must be an int, got {type(diffusion_steps)}")
        num_classes = self.num_classes  # inferred from dataset
        # Create a tensor of labels 0..num_classes-1
        class_labels = tf.range(num_classes, dtype=tf.int32)

        # Generate images conditioned on each label
        generated_images = self.generate(diffusion_steps=diffusion_steps, class_labels=class_labels)

        # Convert to numpy for plotting
        generated_images = generated_images.numpy()

        # Determine grid size (square-ish)
        n_cols = int(np.ceil(np.sqrt(num_classes)))
        n_rows = int(np.ceil(num_classes / n_cols))

        plt.figure(figsize=(n_cols * 3, n_rows * 3))
        for i in range(num_classes):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(generated_images[i])
            plt.title(self.class_names[i])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved generated grid to {save_path}")




# --- TRAINING ---
# create and compile the model
model = DiffusionModel(image_size, widths, block_depth, num_classes=num_classes)
model.class_names = class_names
model.num_classes = num_classes
model.summary()

# below tensorflow 2.9:
# pip install tensorflow_addons
# import tensorflow_addons as tfa
# optimizer=tfa.optimizers.AdamW
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
    loss=keras.losses.mean_absolute_error,
)

# checkpointing (keeps the previous behavior)
checkpoint_dir = "./checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_diffu.weights.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                 save_weights_only=True,
                                                 verbose=1)
'''
checkpoint = tf.train.Checkpoint(model=model)
checkpoint_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: checkpoint.save(str(checkpoint_prefix))
)
'''

def plot_images_callback(epoch, logs):
    save_path = f"./checkpoints/generated_grid_epoch{epoch+1:03d}.png"
    model.plot_images(save_path=save_path, diffusion_steps=20)

plot_callback = keras.callbacks.LambdaCallback(on_epoch_end=plot_images_callback)

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join("./checkpoints/",'logs'), update_freq=1, histogram_freq=1)

try:
    os.mkdir("./checkpoints/")
except OSError as error:
    print("Path already existing: ", error)

try:
    model.build([
    (None, image_size, image_size, 3),  # images
    (None, 1, 1, 1),                    # noise rates
    (None,)                              # class labels
    ])
    model.load_weights(checkpoint_prefix)
except OSError as error:
    print("Loading checkpoint failed : ", error)




# calculate mean and variance of training dataset for normalization
# adapt expects just images -> map to images
model.normalizer.adapt(train_dataset.map(lambda x, y: x))

# run training and plot generated images periodically
model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    callbacks=[plot_callback, checkpoint_callback,tb_callback],
)

# --- INFERENCE (example) ---
# load the last checkpoint and plot images (if desired)
# model.load_weights(checkpoint_prefix)
# model.plot_images()

