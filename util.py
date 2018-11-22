# -----------------------------------------------------------------------
# HashGAN: Deep Learning to Hash with Pair Conditional Wasserstein GAN
# Licensed under The MIT License [see LICENSE for details]
# Modified by Bin Liu
# -----------------------------------------------------------------------

import numpy as np
import locale


# compute param size
def print_param_size(gen_gv, disc_gv):
    print("computing param size")
    for name, grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
        print("{} Params:".format(name))
        total_param_count = 0
        for g, v in grads_and_vars:
            shape = v.get_shape()
            shape_str = ",".join([str(x) for x in v.get_shape()])

            param_count = 1
            for dim in shape:
                param_count *= int(dim)
            total_param_count += param_count

            if g is None:
                print("\t{} ({}) [no grad!]".format(v.name, shape_str))
            else:
                print("\t{} ({})".format(v.name, shape_str))
        print("Total param count: {}".format(
            locale.format("%d", total_param_count, grouping=True)
        ))


class Dataset(object):
    def __init__(self, dataset, output_dim, code_dim):
        print("Initializing Dataset")
        self._dataset = dataset
        self.n_samples = dataset.n_samples
        self._train = dataset.train
        self._output = np.zeros((self.n_samples, output_dim), dtype=np.float32)
        self._codes = np.zeros((self.n_samples, code_dim), dtype=np.float32)

        self._perm = np.arange(self.n_samples)
        np.random.shuffle(self._perm)
        self._index_in_epoch = 0
        self._epochs_complete = 0
        print("Dataset already")
        return

    def next_batch(self, batch_size):
        """
        Args:
          batch_size
        Returns:
          [batch_size, (n_inputs)]: next batch images
          [batch_size, n_class]: next batch labels
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # Another epoch finish
        if self._index_in_epoch > self.n_samples:
            if self._train:
                # Training stage need repeating get batch
                self._epochs_complete += 1
                # Shuffle the data
                np.random.shuffle(self._perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                # Validation stage only process once
                start = self.n_samples - batch_size
                self._index_in_epoch = self.n_samples
        end = self._index_in_epoch

        data, label = self._dataset.data(self._perm[start:end])
        return data, label, self.codes[self._perm[start: end], :]

    def next_batch_output_codes(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # Another epoch finish
        if self._index_in_epoch > self.n_samples:
            if self._train:
                # Shuffle the data
                np.random.shuffle(self._perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                # Validation stage only process once
                start = self.n_samples - batch_size
                self._index_in_epoch = self.n_samples
        end = self._index_in_epoch

        return (self.output[self._perm[start: end], :],
                self.codes[self._perm[start: end], :])

    def feed_batch_output(self, batch_size, output):
        """
        Args:
          batch_size: integer
          output: feature with shape [batch_size, n_output]
        """
        start = self._index_in_epoch - batch_size
        end = self._index_in_epoch
        self.output[self._perm[start:end], :] = output
        return

    def feed_batch_codes(self, batch_size, codes):
        """
        Args:
          batch_size: integer
          codes: binary codes with shape [batch_size, n_output]
        """
        start = self._index_in_epoch - batch_size
        end = self._index_in_epoch
        self.codes[self._perm[start:end], :] = codes
        return

    @property
    def output(self):
        return self._output

    @property
    def codes(self):
        return self._codes

    @property
    def label(self):
        return self._dataset.get_labels

    def finish_epoch(self):
        self._index_in_epoch = 0
        np.random.shuffle(self._perm)


class MAPs:
    def __init__(self, R):
        self.R = R

    @staticmethod
    def distance(a, b):
        return np.dot(a, b)

    def get_mAPs_by_feature(self, database, query):
        ips = np.dot(query.output, database.output.T)
        all_rel = ips
        ids = np.argsort(-all_rel, 1)
        APx = []
        query_labels = query.label
        database_labels = database.label
        print("#calc mAPs# calculating mAPs")
        for i in range(all_rel.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                APx.append(np.sum(Px * imatch) / rel)
        print("mAPs: ", np.mean(np.array(APx)))
        return np.mean(np.array(APx))


class MAPs_CQ:
    def __init__(self, C, subspace_num, subcenter_num, R):
        self.C = C
        self.subspace_num = subspace_num
        self.subcenter_num = subcenter_num
        self.R = R

    def get_mAPs_SQD(self, database, query):
        all_rel = np.dot(np.dot(query.codes, self.C),
                         np.dot(database.codes, self.C).T)
        ids = np.argsort(-all_rel, 1)
        APx = []
        query_labels = query.label
        database_labels = database.label
        # print "#calc mAPs# calculating mAPs"
        for i in range(all_rel.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                APx.append(np.sum(Px * imatch) / rel)
            else:
                APx.append(0.0)
            # if i % 100 == 0:
            # print "step: ", i
        print("SQD mAPs: ", np.mean(np.array(APx)))
        return np.mean(np.array(APx))

    def get_mAPs_AQD(self, database, query):
        all_rel = np.dot(query.output, np.dot(database.codes, self.C).T)
        ids = np.argsort(-all_rel, 1)
        APx = []
        query_labels = query.label
        database_labels = database.label
        # print "#calc mAPs# calculating AQD mAPs"
        for i in range(all_rel.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                APx.append(np.sum(Px * imatch) / rel)
            else:
                APx.append(0.0)
            # if i % 100 == 0:
            # print "step: ", i
        print("AQD mAPs: ", np.mean(np.array(APx)))
        return np.mean(np.array(APx))

    def get_mAPs_by_feature(self, database, query):
        all_rel = np.dot(query.output, database.output.T)
        ids = np.argsort(-all_rel, 1)
        APx = []
        query_labels = query.label
        database_labels = database.label
        # print "#calc mAPs# calculating mAPs"
        for i in range(all_rel.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                APx.append(np.sum(Px * imatch) / rel)
            else:
                APx.append(0.0)
            # if i % 100 == 0:
            # print "step: ", i
        print("Feature mAPs: ", np.mean(np.array(APx)))
        return np.mean(np.array(APx))
