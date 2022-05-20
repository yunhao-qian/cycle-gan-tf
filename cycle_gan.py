from collections import defaultdict
import functools
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _pad_2d(
        tensor: tf.Tensor,
        padding: int,
        mode: Literal['CONSTANT', 'REFLECT', 'SYMMETRIC'],
) -> tf.Tensor:
    return tf.pad(
        tensor,
        tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]]),
        mode,
    )


def _apply_norm(
        inputs: tf.Tensor,
        mode: Literal['batch', 'instance', 'none'],
) -> tf.Tensor:
    if mode == 'batch':
        return layers.BatchNormalization()(inputs)
    elif mode == 'instance':
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        epsilon = 0.001
        return (inputs - mean) / tf.math.sqrt(variance + epsilon)
    elif mode == 'none':
        return inputs
    else:
        raise ValueError(f'unknown norm mode "{mode}"')


def _apply_resnet_block(
        inputs: tf.Tensor,
        filters: int,
        padding_mode: Literal['zeros', 'reflect', 'replicate'],
        norm_mode: Literal['batch', 'instance', 'none'],
        use_dropout: bool,
) -> tf.Tensor:
    conv_use_bias = norm_mode != 'batch'

    def padded_conv(x: tf.Tensor) -> tf.Tensor:
        if padding_mode == 'zeros':
            padding = 'same'
        elif padding_mode == 'reflect':
            x = _pad_2d(x, padding=1, mode='REFLECT')
            padding = 'valid'
        elif padding_mode == 'replicate':
            x = _pad_2d(x, padding=1, mode='SYMMETRIC')
            padding = 'valid'
        else:
            raise ValueError(f'unknown padding mode "{padding_mode}"')
        return layers.Conv2D(
            filters,
            kernel_size=3,
            padding=padding,
            use_bias=conv_use_bias,
        )(x)

    h = padded_conv(inputs)
    h = _apply_norm(h, norm_mode)
    h = layers.ReLU()(h)
    if use_dropout:
        h = layers.Dropout(rate=0.5)(h)
    h = padded_conv(h)
    h = _apply_norm(h, norm_mode)
    return inputs + h


def resnet_generator(
        in_channels: int,
        out_channels: int,
        conv_filters: int = 64,
        norm_mode: Literal['batch', 'instance', 'none'] = 'instance',
        res_use_dropout: bool = False,
        res_blocks: int = 9,
        res_padding_mode: Literal['zeros', 'reflect', 'replicate'] = 'zeros',
) -> keras.Model:
    """
    Creates a ResNet generator model.

    :param in_channels: number of channels of an input image
    :param out_channels: number of channels of an output image
    :param conv_filters: number of filters in the first convolutional layer
    :param norm_mode: type of normalization applied after convolutional layers
    :param res_use_dropout: whether a dropout layer is added between two
        convolutional layers of a ResNet block
    :param res_blocks: number of ResNet blocks
    :param res_padding_mode: padding mode of the convolutional layers of a
        ResNet block
    :return: Resnet generator model that takes an image batch of shape
        `(N, H, W, C_in)` as input and returns an image batch of shape
        `(N, H, W, C_out)` as output
    """

    conv_use_bias = norm_mode != 'batch'
    inputs = keras.Input(shape=(None, None, in_channels))

    h = _pad_2d(inputs, padding=3, mode='REFLECT')
    h = layers.Conv2D(conv_filters, kernel_size=7, use_bias=conv_use_bias)(h)
    h = _apply_norm(h, norm_mode)
    h = layers.ReLU()(h)

    n_downsampling = 2
    for i in range(0, n_downsampling):
        h = layers.Conv2D(
            conv_filters << i + 1,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=conv_use_bias,
        )(h)
        h = _apply_norm(h, norm_mode)
        h = layers.ReLU()(h)

    for _ in range(res_blocks):
        h = _apply_resnet_block(
            h,
            conv_filters << n_downsampling,
            res_padding_mode,
            norm_mode,
            res_use_dropout,
        )

    for i in range(n_downsampling, 0, -1):
        h = layers.Conv2DTranspose(
            conv_filters << i - 1,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=conv_use_bias,
        )(h)
        h = _apply_norm(h, norm_mode)
        h = layers.ReLU()(h)

    h = _pad_2d(h, padding=3, mode='REFLECT')
    h = layers.Conv2D(out_channels, kernel_size=7)(h)
    outputs = tf.math.tanh(h)
    return keras.Model(inputs, outputs)


def n_layer_discriminator(
        in_channels: int,
        conv_filters: int = 64,
        conv_layers: int = 3,
        norm_mode: Literal['batch', 'instance', 'none'] = 'instance',
) -> keras.Model:
    """
    Creates a PatchGAN discriminator model.

    :param in_channels: number of channels of an input image
    :param conv_filters: number of filters in the first convolutional layer
    :param conv_layers: number of downsampling convolutional layers, which
        does not take into account the final convolutional layer that has a
        unitary stride and squeezes the number of channels to one
    :param norm_mode: type of normalization applied after convolutional layers
        except for the first and the last ones
    :return: PatchGAN discriminator model that takes an image batch of shape
        `(N, H_in, W_in, C_in)` as input and returns a probability batch of
        shape `(N, H_out, W_out, 1)` as output
    """

    conv_use_bias = norm_mode != 'batch'
    inputs = keras.Input(shape=(None, None, in_channels))
    h = layers.Conv2D(
        conv_filters,
        kernel_size=4,
        strides=2,
        padding='same',
    )(inputs)
    h = layers.LeakyReLU(alpha=0.2)(h)
    for i in range(1, conv_layers):
        h = layers.Conv2D(
            conv_filters * min(2 ** i, 8),
            kernel_size=4,
            strides=2,
            padding='same',
            use_bias=conv_use_bias,
        )(h)
        h = _apply_norm(h, norm_mode)
        h = layers.LeakyReLU(alpha=0.2)(h)
    h = _pad_2d(h, padding=1, mode='CONSTANT')
    h = layers.Conv2D(
        conv_filters * min(2 ** conv_layers, 8),
        kernel_size=4,
        use_bias=conv_use_bias,
    )(h)
    h = _apply_norm(h, norm_mode)
    h = layers.LeakyReLU(alpha=0.2)(h)
    h = _pad_2d(h, padding=1, mode='CONSTANT')
    outputs = layers.Conv2D(filters=1, kernel_size=4)(h)
    return keras.Model(inputs, outputs)


class ImagePool(layers.Layer):
    """
    `ImagePool` is a helper class that shuffles the images that pass through
    it. On each call, it takes an image patch of shape `(N, H, W, C)` as input
    and returns a shuffled image patch of the same shape.

    Applying an image pool on input images to a discriminator can improve
    training stability, see Section 2.3 of
    https://arxiv.org/pdf/1612.07828.pdf for a detailed explanation.
    """

    def __init__(self, capacity: int = 50):
        """
        Creates an image pool.

        :param capacity: maximum number of images that can be buffered in the
        pool, where a larger capacity leads to shuffling of better quality
        """

        super().__init__()
        self.capacity = tf.constant(capacity)
        self.pool = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        self.pool = tf.Variable(tf.zeros((0, *input_shape[1:])))

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        if not training:
            return inputs

        pool_array = tf.TensorArray(
            dtype=tf.float32,
            size=0,
            dynamic_size=True,
        ).unstack(self.pool)

        output_array = tf.TensorArray(
            dtype=tf.float32,
            size=0,
            dynamic_size=True,
        ).unstack(inputs)

        for i in range(output_array.size()):
            if pool_array.size() < self.capacity:
                pool_array = pool_array.write(pool_array.size(), inputs[i])
            elif tf.random.uniform(shape=[], maxval=2, dtype=tf.int32) == 0:
                j = tf.random.uniform(
                    shape=[], maxval=self.capacity, dtype=tf.int32)
                output_array = output_array.write(i, pool_array.read(j))
                pool_array = pool_array.write(j, inputs[i])

        self.pool.assign(pool_array.stack())
        return output_array.stack()


def _l1_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.math.reduce_mean(tf.math.abs(y_true - y_pred))


def _lsgan_loss(predictions: tf.Tensor, is_real: bool) -> tf.Tensor:
    return tf.math.reduce_mean(tf.math.square(
        predictions - tf.constant(float(is_real))))


def _vanilla_gan_loss(predictions: tf.Tensor, is_real: bool) -> tf.Tensor:
    labels = tf.broadcast_to(tf.constant(float(is_real)), predictions.shape)
    return keras.metrics.binary_crossentropy(
        labels,
        predictions,
        from_logits=True,
        axis=[0, 1, 2, 3],
    )


def _wgangp_loss(predictions: tf.Tensor, is_real: bool) -> tf.Tensor:
    mean = tf.math.reduce_mean(predictions)
    return -mean if is_real else mean


def _save_tensor_as_image(tensor: tf.Tensor, filename: str) -> None:
    array = tensor.numpy()
    if array.ndim == 4:
        array = array.squeeze(axis=0)
    if array.shape[-1] == 1:
        array = array.squeeze(axis=-1)
    array = array * 128 + 127.5
    array = array.clip(min=0, max=255).astype(np.uint8)
    Image.fromarray(array).save(filename)


class CycleGANModel(keras.Model):
    """
    This class represents a CycleGAN model.

    An instance of this class maintains discriminators, generators,
    optimizers, loss functions, and average losses in a training epoch. It can
    either be created from parameters using `CycleGANModel.new()` or loaded
    from saved weights using `keras.Model.load_weights()`.
    """

    def __init__(
            self,
            discriminator_a: keras.Model,
            discriminator_b: keras.Model,
            generator_ab: keras.Model,
            generator_ba: keras.Model,
    ):
        """
        Creates a CycleGAN model from the given discriminators and generators.

        :param discriminator_a: discriminator of domain A
        :param discriminator_b: discriminator of domain A
        :param generator_ab: generator from domain A to domain B
        :param generator_ba: generator from domain B to domain A
        """

        super().__init__()
        self.discriminator_a = discriminator_a
        self.discriminator_b = discriminator_b
        self.generator_ab = generator_ab
        self.generator_ba = generator_ba

        self.optimizer_d = None
        self.optimizer_g = None
        self.fake_a_pool = None
        self.fake_b_pool = None
        self.lambda_a = None
        self.lambda_b = None
        self.lambda_identity = None
        self.gan_loss = None

        self.metric_dict = defaultdict(lambda: keras.metrics.Mean())

    @classmethod
    def new(
            cls,
            a_channels: int,
            b_channels: int,
            norm_mode: Literal['batch', 'instance', 'none'] = 'instance',
            d_conv_filters: int = 64,
            d_conv_layers: int = 3,
            g_conv_filters: int = 64,
            g_res_use_dropout: bool = False,
            g_res_blocks: int = 9,
            g_res_padding_mode: Literal[
                'zeros',
                'reflect',
                'replicate',
            ] = 'zeros',
    ):
        """
        Creates a CycleGAN model whose discriminators and generators are
        constructed from the given parameters.

        :param a_channels: number of channels of an image in domain A
        :param b_channels: number of channels of an image in domain B
        :param norm_mode: normalization mode after convolutional layers in
            discriminators and generators
        :param d_conv_filters: number of filters in the first convolutional
            layer of a discriminator
        :param d_conv_layers: number of downsampling convolutional layers
            (except for the last one) in a discriminator
        :param g_conv_filters: number of filters in the first convolutional
            layer of a generator
        :param g_res_use_dropout: whether a dropout layer is added between two
            convolutional layers of a ResNet block in a generator
        :param g_res_blocks: number of ResNet blocks in a generator
        :param g_res_padding_mode: padding mode of the convolutional layers of
            a ResNet block in a generator
        :return: CycleGAN model
        """

        discriminator_a = n_layer_discriminator(
            a_channels,
            d_conv_filters,
            d_conv_layers,
            norm_mode,
        )
        discriminator_a.compile()
        discriminator_b = n_layer_discriminator(
            b_channels,
            d_conv_filters,
            d_conv_layers,
            norm_mode,
        )
        discriminator_b.compile()
        generator_ab = resnet_generator(
            a_channels,
            b_channels,
            g_conv_filters,
            norm_mode,
            g_res_use_dropout,
            g_res_blocks,
            g_res_padding_mode,
        )
        generator_ab.compile()
        generator_ba = resnet_generator(
            b_channels,
            a_channels,
            g_conv_filters,
            norm_mode,
            g_res_use_dropout,
            g_res_blocks,
            g_res_padding_mode,
        )
        generator_ba.compile()
        return cls(
            discriminator_a,
            discriminator_b,
            generator_ab,
            generator_ba,
        )

    def compile(
            self,
            learning_rate: float = 2e-4,
            adam_beta_1: float = 0.5,
            pool_capacity: int = 50,
            lambda_a: float = 10.0,
            lambda_b: float = 10.0,
            lambda_identity: float = 0.5,
            gan_loss_mode: Literal['lsgan', 'vanilla', 'wgangp'] = 'lsgan',
    ) -> None:
        """
        Compiles loss functions, optimizers, and image pools from the given
        parameters.

        :param learning_rate: learning rate for Adam optimizers
        :param adam_beta_1: beta_1 value for Adam optimizers
        :param pool_capacity: capacity of fake image pools
        :param lambda_a: weight of cycle losses with respect to GAN losses for
            domain A
        :param lambda_b: weight of cycle losses with respect to GAN losses for
            domain B
        :param lambda_identity: weight of identity losses with respect to
            cycle losses
        :param gan_loss_mode: loss function used for GAN losses
        """

        super().compile()
        self.optimizer_d = keras.optimizers.Adam(learning_rate, adam_beta_1)
        self.optimizer_g = keras.optimizers.Adam(learning_rate, adam_beta_1)
        self.fake_a_pool = ImagePool(pool_capacity)
        self.fake_b_pool = ImagePool(pool_capacity)
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.lambda_identity = lambda_identity
        if gan_loss_mode == 'lsgan':
            self.gan_loss = _lsgan_loss
        elif gan_loss_mode == 'vanilla':
            self.gan_loss = _vanilla_gan_loss
        elif gan_loss_mode == 'wgangp':
            self.gan_loss = _wgangp_loss
        else:
            raise ValueError(f'unknown gan loss mode "{gan_loss_mode}"')

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        return list(self.metric_dict.values())

    def _generator_loss(
            self,
            real_a: tf.Tensor,
            real_b: tf.Tensor,
            training: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
        if self.lambda_identity > 0:
            identity_b = self.generator_ab(real_b, training=training)
            loss_identity_ab = _l1_loss(real_b, identity_b) * \
                               self.lambda_b * self.lambda_identity
            identity_a = self.generator_ba(real_a, training=training)
            loss_identity_ba = _l1_loss(real_a, identity_a) * \
                               self.lambda_a * self.lambda_identity
            metrics = {
                'idt_ab': loss_identity_ab,
                'idt_ba': loss_identity_ba,
            }
        else:
            loss_identity_ab = 0
            loss_identity_ba = 0
            metrics = {}

        fake_b = self.generator_ab(real_a, training=training)
        loss_gan_ab = self.gan_loss(
            self.discriminator_b(fake_b, training=training),
            True,
        )
        fake_a = self.generator_ba(real_b, training=training)
        loss_gan_ba = self.gan_loss(
            self.discriminator_a(fake_a, training=training),
            True,
        )
        recons_a = self.generator_ba(fake_b, training=training)
        loss_cycle_a = _l1_loss(real_a, recons_a) * self.lambda_a
        recons_b = self.generator_ab(fake_b, training=training)
        loss_cycle_b = _l1_loss(real_b, recons_b) * self.lambda_b
        generator_loss = loss_identity_ab + loss_identity_ba + \
                         loss_gan_ab + loss_gan_ba + \
                         loss_cycle_a + loss_cycle_b

        metrics.update({
            'gan_ab': loss_gan_ab,
            'gan_ba': loss_gan_ba,
            'cycle_a': loss_cycle_a,
            'cycle_b': loss_cycle_b,
        })
        return generator_loss, fake_a, fake_b, metrics

    def _discriminator_loss(
            self,
            discriminator: keras.Model,
            real: tf.Tensor,
            fake: tf.Tensor,
            training: bool,
    ) -> tf.Tensor:
        loss_real = self.gan_loss(
            discriminator(real, training=training),
            True,
        )
        loss_fake = self.gan_loss(
            discriminator(fake, training=training),
            False,
        )
        return (loss_real + loss_fake) * 0.5

    def train_step(
            self,
            data: Tuple[tf.Tensor, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        real_a, real_b = data
        self.generator_ab.trainable = True
        self.generator_ba.trainable = True
        self.discriminator_a.trainable = False
        self.discriminator_b.trainable = False

        with tf.GradientTape() as tape:
            generator_loss, fake_a, fake_b, metrics = \
                self._generator_loss(real_a, real_b, training=True)
        generator_weights = self.generator_ab.trainable_weights + \
                            self.generator_ba.trainable_weights
        self.optimizer_g.apply_gradients(zip(
            tape.gradient(generator_loss, generator_weights),
            generator_weights,
        ))

        fake_a = self.fake_a_pool(tf.stop_gradient(fake_a))
        fake_b = self.fake_b_pool(tf.stop_gradient(fake_b))
        self.discriminator_a.trainable = True
        self.discriminator_b.trainable = True

        with tf.GradientTape() as tape:
            loss_discriminator_a = self._discriminator_loss(
                self.discriminator_a,
                real_a,
                fake_a,
                training=True,
            )
        gradient_discriminator_a = tape.gradient(
            loss_discriminator_a,
            self.discriminator_a.trainable_weights,
        )
        with tf.GradientTape() as tape:
            loss_discriminator_b = self._discriminator_loss(
                self.discriminator_b,
                real_b,
                fake_b,
                training=True,
            )
        gradient_discriminator_b = tape.gradient(
            loss_discriminator_b,
            self.discriminator_b.trainable_weights,
        )
        self.optimizer_d.apply_gradients(zip(
            gradient_discriminator_a + gradient_discriminator_b,
            self.discriminator_a.trainable_weights +
            self.discriminator_b.trainable_weights,
        ))

        metrics.update({
            'disc_a': loss_discriminator_a,
            'disc_b': loss_discriminator_b,
        })
        for name, value in metrics.items():
            self.metric_dict[name].update_state(value)
        return metrics

    def test_step(
            self,
            data: Tuple[tf.Tensor, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        real_a, real_b = data
        self.generator_ab.trainable = False
        self.generator_ba.trainable = False
        self.discriminator_a.trainable = False
        self.discriminator_b.trainable = False

        generator_loss, fake_a, fake_b, metrics = \
            self._generator_loss(real_a, real_b, training=False)
        loss_discriminator_a = self._discriminator_loss(
            self.discriminator_a,
            real_a,
            fake_a,
            training=False,
        )
        loss_discriminator_b = self._discriminator_loss(
            self.discriminator_b,
            real_b,
            fake_b,
            training=False,
        )

        metrics.update({
            'disc_a': loss_discriminator_a,
            'disc_b': loss_discriminator_b,
        })
        for name, value in metrics.items():
            self.metric_dict[name].update_state(value)
        return metrics

    def save_checkpoint(
            self,
            folder: str,
            examples: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
    ) -> None:
        """
        Save a model checkpoint in the given folder. This includes model
        weights, a JSON file containing average losses, and if example images
        are passed in as input, example output images from generators.

        :param folder: name of the folder to save the checkpoint in
        :param examples: optional pair of real images of shape `(H, W, C)`
            from domain A and B respectively, which are used to generate
            example output images from generators
        """

        os.makedirs(folder, exist_ok=True)

        self.save_weights(os.path.join(folder, 'cycle_gan'))
        self.discriminator_a.save(os.path.join(folder, 'discriminator_a'))
        self.discriminator_b.save(os.path.join(folder, 'discriminator_b'))
        self.generator_ab.save(os.path.join(folder, 'generator_ab'))
        self.generator_ba.save(os.path.join(folder, 'generator_ba'))

        with open(os.path.join(folder, 'metrics.json'), 'w') as fp:
            json.dump({
                name: float(value.result().numpy())
                for name, value in self.metric_dict.items()
            }, fp)

        if examples is None:
            return
        real_a, real_b = examples
        real_a = tf.expand_dims(real_a, axis=0)
        real_b = tf.expand_dims(real_b, axis=0)
        self.generator_ab.trainable = False
        self.generator_ba.trainable = False
        idt_b = self.generator_ab(real_b, training=False)
        idt_a = self.generator_ba(real_a, training=False)
        fake_b = self.generator_ab(real_a, training=False)
        fake_a = self.generator_ba(real_b, training=False)
        recons_a = self.generator_ba(fake_b, training=False)
        recons_b = self.generator_ab(fake_a, training=False)
        _save_tensor_as_image(real_a, os.path.join(folder, 'real_a.png'))
        _save_tensor_as_image(real_b, os.path.join(folder, 'real_b.png'))
        _save_tensor_as_image(idt_a, os.path.join(folder, 'idt_a.png'))
        _save_tensor_as_image(idt_b, os.path.join(folder, 'idt_b.png'))
        _save_tensor_as_image(fake_a, os.path.join(folder, 'fake_a.png'))
        _save_tensor_as_image(fake_b, os.path.join(folder, 'fake_b.png'))
        _save_tensor_as_image(recons_a, os.path.join(folder, 'recons_a.png'))
        _save_tensor_as_image(recons_b, os.path.join(folder, 'recons_b.png'))


def linear_lr_schedule(
        epochs_decay: int,
) -> Callable[[int, float, int], float]:
    """
    Creates a linear learning-rate schedule. In this schedule, the learning
    rate is constant in the beginning epochs and decays linearly to zero in
    the last `epochs_decay` epochs.

    :param epochs_decay: number of epochs in which the learning rate is
        decaying
    :return: callable that takes `(epoch, lr, num_epochs)` as input and
        returns the updated learning rate as output
    """

    def schedule(epoch: int, lr: float, epochs: int) -> float:
        if epochs - epochs_decay < epoch < epochs:
            return lr * ((epochs - epoch) / (epochs - epoch + 1))
        else:
            return lr
    return schedule


def step_lr_schedule(
        step_size: int,
        gamma: float = 0.1,
) -> Callable[[int, float, int], float]:
    """
    Creates a step learning-rate schedule. In this schedule, the learning rate
    decays by a factor of `gamma` for every `step_size` epochs.

    :param step_size: number of epochs by which a decay in learning rate
        happens
    :param gamma: decay factor of the learning rate
    :return: callable that takes `(epoch, lr, num_epochs)` as input and
        returns the updated learning rate as output
    """

    def schedule(epoch: int, lr: float, _: int) -> float:
        if epoch != 0 and epoch % step_size == 0:
            return lr * gamma
        else:
            return lr
    return schedule


def cosine_annealing_lr_schedule() -> Callable[[int, float, int], float]:
    """
    Creates a cosine-annealing learning rate schedule. In this schedule, the
    learning rate decays to zero following a cosine-shaped path.

    :return: callable that takes `(epoch, lr, num_epochs)` as input and
        returns the updated learning rate as output
    """

    def schedule(epoch: int, lr: float, epochs: int) -> float:
        if 0 < epoch < epochs:
            return lr * (1 + np.cos(np.pi * (epoch / epochs))) / \
                        (1 + np.cos(np.pi * ((epoch - 1) / epochs)))
        else:
            return lr
    return schedule


class LearningRateScheduler(keras.callbacks.Callback):
    """
    This class is a Keras callback that schedules learning-rate updates at the
    beginning of every epoch.
    """

    def __init__(self, schedule: Callable[[int, float, int], float]):
        """
        Creates a learning rate scheduler.

        :param schedule: callable that takes `(epoch, lr, num_epochs)` as
            input and returns the updated learning rate as output
        """

        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch: int, _):
        old_d_lr = keras.backend.get_value(self.model.optimizer_d.lr)
        new_d_lr = self.schedule(epoch, old_d_lr, self.params['epochs'])
        keras.backend.set_value(self.model.optimizer_d.lr, new_d_lr)

        old_g_lr = keras.backend.get_value(self.model.optimizer_g.lr)
        new_g_lr = self.schedule(epoch, old_g_lr, self.params['epochs'])
        keras.backend.set_value(self.model.optimizer_g.lr, new_g_lr)

        if self.params['verbose'] != 0:
            tf.print(
                f'learning rate ({old_d_lr}, {old_g_lr}) '
                f'-> ({new_d_lr}, {new_g_lr})',
                output_stream=sys.stdout,
            )


class ModelCheckpoint(keras.callbacks.Callback):
    """
    This class is a Keras callback that saves a checkpoint of the model at the
    end of every epoch.
    """

    def __init__(
            self,
            folder: str,
            validation_data: Optional[tf.data.Dataset] = None,
    ):
        """
        Creates a model checkpoint callback.

        :param folder: name of the folder to save checkpoints in
        :param validation_data: optional dataset that, on each iteration,
        returns a pair of real images of shape `(H, W, C)` from domain A and B
        respectively, which is used to generate example output images from
        generators
        """

        super().__init__()
        self.folder = folder
        self.validation_data = validation_data

    def on_epoch_end(self, epoch: int, _):
        epoch_folder = os.path.join(self.folder, f'epoch-{epoch:04d}')
        if self.validation_data is None:
            examples = None
        else:
            examples = next(iter(self.validation_data))
        self.model.save_checkpoint(epoch_folder, examples)


def _transform_dataset(
        item: Dict[str, tf.Tensor],
        load_size: Tuple[int, int],
        crop_size: Tuple[int, int],
) -> tf.Tensor:
    image = tf.image.resize(item['image'], tf.constant(load_size))
    image = tf.image.random_crop(
        image,
        tf.constant(crop_size + (image.shape[-1],)),
    )
    image = (tf.cast(image, tf.float32) - 127.5) / 128.0
    return image


def preprocess_datasets(
        train_a: tf.data.Dataset,
        train_b: tf.data.Dataset,
        test_a: tf.data.Dataset,
        test_b: tf.data.Dataset,
        load_size: Tuple[int, int] = (286, 286),
        crop_size: Tuple[int, int] = (256, 256),
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Preprocesses image datasets so that images can be fed into a CycleGAN
    model.

    An input dataset, on each iteration, returns a dictionary with key
    "image" mapped to an uint8 tensor of shape `(H, W, C)`. An output dataset,
    on each iteration, returns a pair of float32 tensors of shape `(H, W, C)`
    from domain A and B respectively, where pixel values of output tensors are
    scaled into (-1, 1).

    Size of an input image is first rescaled into `load_size` (in which the
    aspect ratio is not respected), and then randomly cropped into
    `crop_size`.

    :param train_a: training set of domain A
    :param train_b: training set of domain B
    :param test_a: test set of domain A
    :param test_b: test set of domain B
    :param load_size: size that an original image is rescaled into
    :param crop_size: size that a rescaled image is randomly cropped into
    :return: tuple of the training set and the test set
    """

    transform = functools.partial(
        _transform_dataset,
        load_size=load_size,
        crop_size=crop_size,
    )
    train_set = tf.data.Dataset.zip((
        train_a.map(transform),
        train_b.map(transform),
    ))
    test_set = tf.data.Dataset.zip((
        test_a.map(transform),
        test_b.map(transform),
    ))
    return train_set, test_set
