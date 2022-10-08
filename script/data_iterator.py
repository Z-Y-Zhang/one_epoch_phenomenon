import json
import cPickle as pkl
import random
import numpy as np
import os
import time
import traceback
import sklearn
import pandas as pd

class DataIterator:

    def __init__(self, source,
                 batch_size=128,
                 maxlen=100,
                 skip_empty=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 minlen=None,
                 parall=False,
                 data_in_memory=False,
                 max_samples=0,
                 args=None
                ):
        self.source = open(source, 'r')
        #self
        self.source_dicts = []

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.args = args
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        #self.k = batch_size * max_batch_size
        self.k = batch_size

        self.end_of_data = False


        self.lines = []
        # read file into self.lines
        if  max_samples>0 or data_in_memory:
            print "data_in_memory = True, reading data..."
            self.lines = []
            for line in self.source.readlines():
                # print line
                self.lines.append(line)
            if max_samples>0 and max_samples<len(self.lines):
                step = len(self.lines) // max_samples
                lines = []
                for i in range(0, len(self.lines), step):
                    lines.append(self.lines[i])

                self.lines = lines


        self.line_index = 0
    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        if self.lines:
            self.line_index = 0

    def shuffle(self, seed=0):
        # shuffle every
        if self.lines:
            print "shuffle, random seed:", seed
            self.lines = sklearn.utils.shuffle(self.lines, random_state=seed)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        hist_item_list = []
        hist_cate_list = []

        neg_item_list = []
        neg_cate_list = []

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                if self.lines:
                    ss = self.lines[self.line_index]
                    self.line_index += 1
                else:
                    ss = self.source.readline()
                # print ss
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break

                uid = int(ss[0])
                item_id = int(ss[1])
                cate_id = int(ss[2])
                label = int(ss[3])

                hist_item = map(int, ss[4].split(","))
                hist_cate = map(int, ss[5].split(","))

                neg_item = map(int, ss[6].split(","))
                neg_cate = map(int, ss[7].split(","))

                source.append([uid, item_id, cate_id])
                target.append([label])
                hist_item_list.append(hist_item[-self.maxlen:])
                hist_cate_list.append(hist_cate[-self.maxlen:])

                neg_item_list.append(neg_item[-self.maxlen:])
                neg_cate_list.append(neg_cate[-self.maxlen:])


                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0 or len(source) < self.batch_size:
            source, target = self.next()

        uid_array = np.array(source)[:,0]
        item_array = np.array(source)[:,1]
        cate_array = np.array(source)[:,2]

        target_array = np.array(target)

        history_item_array = np.array(hist_item_list)
        history_cate_array = np.array(hist_cate_list)

        history_neg_item_array = np.array(neg_item_list)
        history_neg_cate_array = np.array(neg_cate_list)

        history_mask_array = np.greater(history_item_array, 0)*1.0
        if self.args and self.args.corruption_percent > 0:
            hash_key = pd.util.hash_array(uid_array+item_array+cate_array+np.sum(history_item_array, axis=1)+np.sum(history_cate_array, axis=1))%10000
            hash_mask = hash_key < (self.args.corruption_percent*10000)
            y_hat = hash_key%2 #label.shape = [batch, 1]
            target_array[:,0][hash_mask] = y_hat[hash_mask]

        if self.args and self.args.filter_percent >= 0:
            filter_list = [uid_array, item_array, cate_array, history_item_array, history_cate_array, history_neg_item_array, history_neg_cate_array]
            n_T= int(self.args.n_id*self.args.filter_percent)

            for array in filter_list:
                array[array>n_T] = 1 #default value

        if self.args and self.args.rehash >= 0:
            rehash_list = [uid_array, item_array, cate_array, history_item_array, history_cate_array, history_neg_item_array, history_neg_cate_array]
            n_id = int(self.args.n_id*self.args.rehash)
            [uid_array, item_array, cate_array, history_item_array, history_cate_array, history_neg_item_array, history_neg_cate_array] = [pd.util.hash_array(array)%n_id + 1 for array in  rehash_list]
            # set None value to 0
            history_item_array[np.less(history_mask_array, 1)] = 0
            history_cate_array[np.less(history_mask_array, 1)] = 0

        return (uid_array, item_array, cate_array), (target_array, history_item_array, history_cate_array, history_neg_item_array, history_neg_cate_array, history_mask_array)

