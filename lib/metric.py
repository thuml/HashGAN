import numpy as np


class MAPs:
    def __init__(self, r):
        self.R = r

    @staticmethod
    def distance(a, b):
        return np.dot(a, b)

    def get_maps_by_feature(self, database, query):
        ips = np.dot(query.output, database.output.T)
        ids = np.argsort(-ips, 1)
        apx = []
        for i in range(ips.shape[0]):
            label = query.label[i, :].copy()
            label[label == 0] = -1
            imatch = np.sum(database.label[ids[i, :][0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            px = np.cumsum(imatch).astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                apx.append(np.sum(px * imatch) / rel)
        return np.mean(np.array(apx))
