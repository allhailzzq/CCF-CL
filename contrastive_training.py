import tensorflow as tf
import time
import os
import numpy as np
import tensorflow_probability as tfp
import argparse


class Config(object):
    def __init__(self):
        self.embedding_dim = 128
        self.batchsize = batchsize


class FeatureNet(tf.keras.Model):
    def __init__(self, n_code):
        super(FeatureNet, self).__init__()

        self.encoder = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=n + 1,
                                                                                   output_mode='binary')
        self.linear = tf.keras.layers.Dense(config.embedding_dim, activation=tf.nn.relu)
        self.lstm = tf.keras.layers.LSTM(config.embedding_dim, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
        self.mlp0 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.mlp1 = tf.keras.layers.Dense(256)
        self.n = n_code

    def call(self, code, aux_info, length, is_training):
        if self.n > 0:
            code = tf.reshape(code, [batchsize*max_num_visit,-1])
            x = self.linear(tf.concat((tf.reshape(self.encoder(code)[:,1:], [batchsize, max_num_visit, self.n]),aux_info),axis=-1))
        else:
            x = self.linear(aux_info)
        x = self.lstm(x, training=is_training)
        x = tf.gather_nd(x, tf.concat(
            (tf.expand_dims(tf.range(batchsize, dtype=tf.int32), -1), tf.expand_dims(length - 1, -1)), axis=-1))
        feature_vec = self.mlp1(self.mlp0(x))
        feature_vec = feature_vec / tf.math.sqrt(tf.reduce_sum(feature_vec ** 2, axis=-1, keepdims=True))
        return feature_vec


def train():
    beta1 = tfd.Beta(2, 4)
    beta2 = tfd.Beta(2, 4)
    
    # data augmentation to generate views
    def myfunc(feature, aux_info, length):
        length = tf.cast(length, tf.float32)
        length1 = tf.cast(beta1.sample() * (length - 5) + 5, tf.int32)
        length2 = tf.cast(beta2.sample() * (length - 5) + 5, tf.int32)
        length = tf.cast(length, tf.int32)
        pos1 = tf.random.uniform(shape=(), minval=0, maxval=length - length1 + 1, dtype=tf.int32)
        pos2 = tf.random.uniform(shape=(), minval=0, maxval=length - length2 + 1, dtype=tf.int32)
        feature1 = tf.concat(
            (feature[pos1:pos1 + length1], tf.zeros((200 - length1, 50), dtype=tf.int32)), axis=0)
        feature2 = tf.concat(
            (feature[pos2:pos2 + length2], tf.zeros((200 - length2, 50), dtype=tf.int32)), axis=0)
        
        # the aux_info has 39 dimensions
        aux_info1 = tf.concat((aux_info[pos1:pos1 + length1], tf.zeros((200 - length1, 39), dtype=tf.float32)), axis=0)
        aux_info2 = tf.concat((aux_info[pos2:pos2 + length2], tf.zeros((200 - length2, 39), dtype=tf.float32)), axis=0)
        return (feature1, aux_info1, length1), \
               (tf.zeros_like(feature2), aux_info2, length2), \
               (tf.where(feature2 < 244 + 1, feature2, tf.zeros_like(feature2)), aux_info2[:, 19:], length2), \
               (tf.where(feature2 >= 244 + 1, feature2 - 244, tf.zeros_like(feature2)), aux_info2[:, 19:], length2)

    lengths = np.load('length.npy').astype('int32') # the number of visits for each patient
    features = np.load('code.npy').astype('int32')[:,:,:50] + 1 # only keep the first 50 codes
    aux_infos = np.load('aux_info.npy').astype('float32')
    train_idx, val_idx = np.load('train_idx.npy'), np.load('val_idx.npy')
    dataset_train = tf.data.Dataset.from_tensor_slices((features[train_idx], aux_infos[train_idx],
                                                        lengths[train_idx])).shuffle(4096 * 2,
                                                                                     reshuffle_each_iteration=True)
    parsed_dataset_train = dataset_train.map(myfunc, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
        batchsize, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    dataset_val = tf.data.Dataset.from_tensor_slices((features[val_idx], aux_infos[val_idx],
                                                      lengths[val_idx])).map(myfunc,
                                                                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
    parsed_dataset_val = dataset_val.batch(batchsize, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    del features, aux_infos
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    feature_net = FeatureNet(526)
    
    # local_1 only has other information, local_2 only has procedure, local_3 only has diagnosis
    local_1 = FeatureNet(0)
    local_2 = FeatureNet(244)
    local_3 = FeatureNet(282)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, feature_net=feature_net, local_1=local_1, local_2=local_2, local_3=local_3)
    print('start')

    @tf.function
    def one_step(batch, batch_1, batch_2, batch_3, is_training):
        # contrastive learning loss
        with tf.GradientTape() as tape:
            feature_vec = feature_net(*batch, is_training)
            feature_vec_1 = local_1(*batch_1, is_training)
            feature_vec_2 = local_2(*batch_2, is_training)
            feature_vec_3 = local_3(*batch_3, is_training)
            pair_wise_1 = tf.matmul(feature_vec, feature_vec_1, transpose_b=True) * 10
            loss1 = tf.linalg.diag_part(tf.nn.log_softmax(pair_wise_1))
            loss1 = -tf.reduce_mean(loss1 * (1 - tf.math.exp(loss1)) ** 2)
            pair_wise_2 = tf.matmul(feature_vec, feature_vec_2, transpose_b=True) * 10
            loss2 = tf.linalg.diag_part(tf.nn.log_softmax(pair_wise_2))
            loss2 = -tf.reduce_mean(loss2 * (1 - tf.math.exp(loss2)) ** 2)
            pair_wise_3 = tf.matmul(feature_vec, feature_vec_3, transpose_b=True) * 10
            loss3 = tf.linalg.diag_part(tf.nn.log_softmax(pair_wise_3))
            loss3 = -tf.reduce_mean(loss3 * (1 - tf.math.exp(loss3)) ** 2)
            loss = loss1 + loss2 + loss3
        if is_training:
            grads = tape.gradient(loss,
                                  feature_net.trainable_variables + local_1.trainable_variables + local_2.trainable_variables + local_3.trainable_variables)
            optimizer.apply_gradients(zip(grads, feature_net.trainable_variables + local_1.trainable_variables + local_2.trainable_variables + local_3.trainable_variables))
        return loss

    print('training start')
    for epoch in range(800):
        step_val = 0
        step_train = 0

        start_time = time.time()
        loss_val = 0
        loss_train = 0

        for batch_sample in parsed_dataset_train:
            aug1, aug2, aug3, aug4 = batch_sample
            step_loss = one_step(aug1, aug2, aug3, aug4, True).numpy()
            loss_train += step_loss
            step_train += 1

        for batch_sample in parsed_dataset_val:
            aug1, aug2, aug3, aug4 = batch_sample
            step_loss = one_step(aug1, aug2, aug3, aug4, False).numpy()
            loss_val += step_loss
            step_val += 1

        duration_epoch = int(time.time() - start_time)
        format_str = 'epoch: %d, train_loss = %f, val_loss = %f (%d)'
        print(format_str % (epoch, loss_train / step_train, loss_val / step_val,
                            duration_epoch))
        if epoch % 20 == 19:
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    tfd = tfp.distributions
    batchsize = 128
    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=str)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

    max_code_visit = 50
    max_num_visit = 200

    checkpoint_directory = "training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    train()
