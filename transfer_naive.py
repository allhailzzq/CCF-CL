import tensorflow as tf
import time
import os
import numpy as np
import argparse


class Config(object):
    def __init__(self):
        self.embedding_dim = 128
        self.batchsize = batchsize


class FeatureNet(tf.keras.Model):
    def __init__(self):
        super(FeatureNet, self).__init__()

        self.encoder = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=526 + 1,
                                                                                   output_mode='binary')
        self.linear = tf.keras.layers.Dense(config.embedding_dim, activation=tf.nn.relu)
        self.lstm = tf.keras.layers.LSTM(config.embedding_dim, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
        self.mlp0 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.mlp1 = tf.keras.layers.Dense(256)

    def call(self, code, aux_info, length, is_training):
        n = code.shape[0]
        code = tf.reshape(code, [n*max_num_visit,-1])
        x = self.linear(tf.concat((tf.reshape(self.encoder(code)[:,1:], [n, max_num_visit, 526]),aux_info),axis=-1))
        x = self.lstm(x, training=is_training)
        x = tf.gather_nd(x, tf.concat(
            (tf.expand_dims(tf.range(n, dtype=tf.int32), -1), tf.expand_dims(length - 1, -1)), axis=-1))
        feature_vec = self.mlp1(self.mlp0(x))
        feature_vec = feature_vec / tf.math.sqrt(tf.reduce_sum(feature_vec ** 2, axis=-1, keepdims=True))
        return feature_vec


class PredNet(tf.keras.Model):
    def __init__(self):
        super(PredNet, self).__init__()
        self.dense = tf.keras.layers.Dense(120, activation=tf.nn.sigmoid)

    def call(self, latent):
        output = self.dense(latent)
        return output


def load():
    feature_net = FeatureNet()
    checkpoint = tf.train.Checkpoint(feature_net=feature_net)
    checkpoint.restore(checkpoint_prefix + '-' + args.epoch).expect_partial()

    @tf.function
    def train_step(batch):
        feature_vec = feature_net(*batch, False)
        return feature_vec

    lengths = np.load('length.npy').astype('int32')
    features = np.load('code.npy').astype('int32')[:,:,:50] + 1
    aux_infos = np.load('aux_info.npy').astype('float32')
    train_idx, val_idx, test_idx = np.load('train_idx.npy'), np.load('val_idx.npy'), \
                                   np.load('test_idx.npy')

    dataset_train = tf.data.Dataset.from_tensor_slices((features[train_idx],
                                                        aux_infos[train_idx],
                                                        lengths[train_idx]))
    parsed_dataset_train = dataset_train.batch(batchsize, drop_remainder=False)
    outputs = []
    for batch_sample in parsed_dataset_train:
        outputs.extend(train_step(batch_sample).numpy())
    latent_dic = np.array(outputs)
    np.save('latent_train_' + args.model + '_naive', latent_dic)

    dataset_train = tf.data.Dataset.from_tensor_slices((features[test_idx],
                                                        aux_infos[test_idx],
                                                        lengths[test_idx]))
    parsed_dataset_train = dataset_train.batch(batchsize, drop_remainder=False)
    outputs = []
    for batch_sample in parsed_dataset_train:
        outputs.extend(train_step(batch_sample).numpy())
    latent_dic = np.array(outputs)
    np.save('latent_test_' + args.model + '_naive', latent_dic)

    dataset_train = tf.data.Dataset.from_tensor_slices((features[val_idx],
                                                        aux_infos[val_idx],
                                                        lengths[val_idx]))
    parsed_dataset_train = dataset_train.batch(batchsize, drop_remainder=False)
    outputs = []
    for batch_sample in parsed_dataset_train:
        outputs.extend(train_step(batch_sample).numpy())
    latent_dic = np.array(outputs)
    np.save('latent_val_' + args.model + '_naive', latent_dic)
    return


def main():
    def train():
        pred_net = PredNet()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        m1 = [tf.keras.metrics.AUC(num_thresholds=200) for _ in range(120)]
        m2 = [tf.keras.metrics.AUC(num_thresholds=200, curve='PR') for _ in range(120)]
        encoder = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=284, output_mode='binary')

        @tf.function
        def one_step(latent, target, target_range, is_training):
            target = tf.gather(encoder(target + 1)[:, 1:], weight_idx, axis=1)
            target_range = tf.gather(1 - encoder(target_range + 1)[:, 1:], weight_idx, axis=1)
            with tf.GradientTape() as tape:
                output = pred_net(latent)
                loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(target, output) * target_range,axis=0) / tf.reduce_sum(
                    target_range,axis=0)
                loss_opt = tf.reduce_mean(loss)
            if is_training:
                grads = tape.gradient(loss_opt, pred_net.trainable_variables)
                optimizer.apply_gradients(zip(grads, pred_net.trainable_variables))
            else:
                for k in range(120):
                    update_target = tf.boolean_mask(target[:, k], tf.cast(target_range, tf.bool)[:, k])
                    update_output = tf.boolean_mask(output[:, k], tf.cast(target_range, tf.bool)[:, k])
                    m1[k].update_state(update_target, update_output)
                    m2[k].update_state(update_target, update_output)
            return loss

        min_val_loss = [10000]*120
        max_auc = [0] * 120
        max_prc = [0] * 120
        print('training start')
        patience = [0] * 120
        t = time.time()
        for epoch in range(1000):
            loss_train = []
            for batch in parsed_dataset_train:
                loss_train.append(one_step(*batch, True).numpy())

            loss_val = []
            for batch in parsed_dataset_val:
                loss_val.append(one_step(*batch, False).numpy())

            loss_test = []
            for mi in m1:
                mi.reset_states()
            for mi in m2:
                mi.reset_states()
            for batch in parsed_dataset_test:
                loss_test.append(one_step(*batch, False))
            auc_test = np.array([mi.result().numpy() for mi in m1])
            prc_test = np.array([mi.result().numpy() for mi in m2])

            if epoch % 10 == 9:
                print(format_str % (epoch, np.mean(loss_train),
                                    np.mean(loss_val), np.mean(auc_test), np.mean(prc_test),
                                    (time.time() - t) / 60))
                t = time.time()
            for i in range(120):
                tmp_loss = np.mean(loss_val,axis=0)[i]
                if patience[i] < 15 or epoch < 15:
                    if tmp_loss < min_val_loss[i]:
                        patience[i] = 0
                        min_val_loss[i] = tmp_loss
                        max_auc[i] = auc_test[i]
                        max_prc[i] = prc_test[i]
                    else:
                        patience[i] += 1
            if np.sum(np.array(patience) >= 15) == 120:
                break
        return max_auc, max_prc

    targets = np.load('target.npy').astype('int32')
    histories = np.load('history.npy').astype('int32')
    train_idx, test_idx, val_idx = np.load('train_idx.npy'), np.load('test_idx.npy'), \
                                   np.load('val_idx.npy')
    latent_train_dic, latent_test_dic, latent_val_dic = np.load('latent_train_' + args.model + '_naive.npy'), np.load('latent_test_' + args.model + '_naive.npy'), \
                                                        np.load('latent_val_' + args.model + '_naive.npy')
    dataset_train = tf.data.Dataset.from_tensor_slices((latent_train_dic,
                                                        targets[train_idx],
                                                        histories[train_idx])).shuffle(4096 * 3,
                                                                                       reshuffle_each_iteration=True)
    parsed_dataset_train = dataset_train.batch(192, drop_remainder=True)

    dataset_test = tf.data.Dataset.from_tensor_slices((latent_test_dic,
                                                       targets[test_idx],
                                                       histories[test_idx]))
    parsed_dataset_test = dataset_test.batch(batchsize, drop_remainder=True)

    dataset_val = tf.data.Dataset.from_tensor_slices((latent_val_dic,
                                                      targets[val_idx],
                                                      histories[val_idx]))
    parsed_dataset_val = dataset_val.batch(batchsize, drop_remainder=True)
    # auc_ = []
    # prc_ = []
    # for i in range(2):
    auc, prc = train()
    print(np.mean(auc), np.mean(prc))
        # auc_.append(auc)
        # prc_.append(prc)
    np.save('result_naive/auc_' + args.model, auc)
    np.save('result_naive/prc_' + args.model, prc)


if __name__ == '__main__':
    batchsize = 32
    max_code_visit = 50
    max_num_visit = 200
    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('epoch', type=str)
    parser.add_argument('load', type=bool)
    args = parser.parse_args()
    checkpoint_directory = 'training_checkpoints_' + args.model + '_naive'
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

    format_str = 'epoch: %d, train_loss = %f, val_loss = %f, test_auc = %f, test_prc = %f (%d)'

    # find the codes with more than 50 occurences as prediction target
    weight_idx = tf.constant(np.arange(283)[np.load('label_weight.npy') >= 50], dtype=tf.int32)
    if args.load:
        load()
    main()
