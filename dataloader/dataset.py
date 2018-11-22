# =============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import math
import cv2
import numpy as np


class Dataset(object):
    def __init__(self, path, train=True, height_width=256):
        self.lines = open(path, 'r').readlines()
        self.n_samples = len(self.lines)
        self.train = train
        self.height_width = height_width
        self.img_shape = (self.height_width, self.height_width)

        self._img = [0] * self.n_samples
        self._label = [0] * self.n_samples
        self._load = [0] * self.n_samples
        self._load_num = 0
        self._status = 0
        self.data = self.img_data
        self.all_data = self.img_all_data

    def read_image_at(self, index):
        filename = self.lines[index].strip().split()[0]
        img = cv2.imread(filename)
        return cv2.resize(img, self.img_shape, interpolation=cv2.INTER_AREA)

    def get_label(self, index):
        return [int(j) for j in self.lines[index].strip().split()[1:]]

    def img_data(self, index):
        if self._status:
            return self._img[index, :], self._label[index, :]
        else:
            ret_img = []
            ret_label = []
            for i in index:
                # noinspection PyBroadException
                try:
                    if self.train:
                        if not self._load[i]:
                            self._img[i] = self.read_image_at(i)
                            self._label[i] = self.get_label(i)
                            self._load[i] = 1
                            self._load_num += 1
                        ret_img.append(self._img[i])
                        ret_label.append(self._label[i])
                    else:
                        self._label[i] = self.get_label(i)
                        ret_img.append(self.read_image_at(i))
                        ret_label.append(self._label[i])
                except:
                    print('cannot open', self.lines[i])

            if self._load_num == self.n_samples:
                self._status = 1
                self._img = np.asarray(self._img)
                self._label = np.asarray(self._label)
            return np.asarray(ret_img), np.asarray(ret_label)

    @property
    def img_all_data(self):
        if self._status:
            return self._img, self._label
        else:
            return None, None

    def get_labels(self):
        for i in range(self.n_samples):
            if self._label[i] == 0:
                self._label[i] = self.get_label(i)
        return np.asarray(self._label)


def data_generator(batch_size, width_height, file_name):
    _dataset = Dataset(file_name, True, width_height)

    def get_epoch():

        _index_in_epoch = 0
        _perm = np.arange(_dataset.n_samples)
        np.random.shuffle(_perm)
        for _ in range(int(math.ceil(_dataset.n_samples / batch_size))):
            start = _index_in_epoch
            _index_in_epoch += batch_size
            # finish one epoch
            if _index_in_epoch > _dataset.n_samples:
                data, label = _dataset.data(_perm[start:])
                data1, label1 = _dataset.data(
                    _perm[:_index_in_epoch - _dataset.n_samples])
                data = np.concatenate([data, data1], axis=0)
                label = np.concatenate([label, label1], axis=0)
            else:
                end = _index_in_epoch
                data, label = _dataset.data(_perm[start:end])

            # n*h*w*c -> n*c*h*w
            data = np.transpose(data, (0, 3, 1, 2))
            # bgr -> rgb
            data = data[:, ::-1, :, :]
            data = np.reshape(data, (batch_size, -1))
            yield (data, label)

    return get_epoch


def load_train(batch_size, width_height, data_root):
    return [data_generator(batch_size, width_height, os.path.join(data_root, split + '.txt'))
            for split in ["train", "database_nolabel", "test"]]

def load_val(batch_size, width_height, data_root):
    return [data_generator(batch_size, width_height, os.path.join(data_root, split + '.txt'))
            for split in ["database", "test.txt"]]

