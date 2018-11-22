import numpy as np


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
