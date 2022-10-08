import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
from utils import *
import multiprocessing
import argparse
import cPickle as pkl
import numpy as np
import pandas as pd
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001,  help='learning rate')
parser.add_argument('--epochs', type=int, default=1,  help='train epochs')
parser.add_argument('--batch_size', type=int, default=128,  help='batch size')

parser.add_argument('--embed_dim', type=int, default=8,  help='embedding dim')
parser.add_argument('--load_pretrain', type=int, default=0, help='load the pretrained model of x iter. 0: do not load')
parser.add_argument('--train_para', type=str, default='all', choices=['all', 'emb', 'fc', 'fc3'])
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--opt_decay', type=float, default=0.9, help='decay rate of adam/rmsprop, default=0.9')
parser.add_argument('--iter', type=int, default=99999999, help='max train iter')
parser.add_argument('--save_emb', type=int, default=0, choices=[0,1], help='wheather to save embedding feature vector, if yes, then only run 1 iter')
parser.add_argument('--model_type', type=str, default='DNN')
parser.add_argument('--corruption_percent', type=float, default=0, help='0.0 <= percent <= 1.0')
parser.add_argument('--whole_train_set', type=int, default=1, help='whether to use whole train set to estimate train loss')
parser.add_argument('--weight_decay', type=float, default=0, help='whether to use weight decay, lr of weight decay (e.g., 0.001)')
parser.add_argument('--data_shuffle', type=str, default='every', help='whether to shuffle data. no | once | every|')
parser.add_argument('--rehash', type=float, default=-1, help='re-hash: precent of ids, 0<rehash<=1')

parser.add_argument('--print_grad', type=int, default=0 )
parser.add_argument('--filter_percent', type=float, default=-1.0)
parser.add_argument('--save_model', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--neuron', type=float, default=1, help=' 0<neuron')
parser.add_argument('--nlayers', type=int, default=2, help='nlayers>=1')
parser.add_argument('--n_id', type=int, default=426000, help='number of ids')
parser.add_argument('--test_iter', type=int, default=50)
parser.add_argument('--activation', type=str, default='dice', help='activation function. dice | sigmoid | relu| prelu')
parser.add_argument('--dataset', type=str, default='book')




args = parser.parse_args()


def generator_queue(generator, max_q_size=20,
                    wait_time=0.1, nb_worker=1):
    generator_threads = []
    q = multiprocessing.Queue(maxsize=max_q_size)
    _stop = multiprocessing.Event()
    try:
        def data_generator_task():
            while not _stop.is_set():
                try:
                    if q.qsize() < max_q_size:
                        generator_output = next(generator)
                        q.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception:
                    _stop.set()
                    print("over1")
                    #raise

                # ##debug start####
                #
                # if q.qsize() < max_q_size:
                #     #start_time = time.time()
                #     generator_output = next(generator)
                #     #end_time = time.time()
                #     #print end_time - start_time
                #     q.put(generator_output)
                # else:
                #     time.sleep(wait_time)
                #
                # ###debug end######

        for i in range(nb_worker):
            thread = multiprocessing.Process(target=data_generator_task)
            generator_threads.append(thread)
            thread.daemon = True
            thread.start()
    except Exception:
        _stop.set()
        for p in generator_threads:
            if p.is_alive():
                p.terminate()
        q.close()

    return q, _stop, generator_threads

EMBEDDING_DIM = args.embed_dim
HIDDEN_SIZE = 8 * 2
best_auc = 0.0

def prepare_data(src, target):
    nick_id, item_id, cate_id = src
    label, hist_item, hist_cate, neg_item, neg_cate, hist_mask = target
    return nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask

def eval(sess, test_data, model, model_path, batch_size, is_batch_auc=False, iter=-1):
    loss_sum = 0.
    loss_list = []
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    test_data_pool, _stop, _ = generator_queue(test_data)
    while True:
        if  _stop.is_set() and test_data_pool.empty():
            break
        if not test_data_pool.empty():
            src,tgt = test_data_pool.get()
        else:
            continue
        nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask = prepare_data(src, tgt)
        if len(nick_id) < batch_size:
            continue
        nums += 1
        target = label
        prob, loss, acc, aux_loss, cal_dict = model.calculate(sess, [nick_id, item_id, cate_id, hist_item, hist_cate, neg_item, neg_cate, hist_mask, label])
        loss_sum += loss
        loss_list.append(loss)
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p ,t in zip(prob_1, target_1):
            stored_arr.append([p, t])
    # print "loss_list=",loss_list
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        #model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum

def train(
        train_file = "./data/book_data/book_train.txt",
        test_file = "./data/book_data/book_test.txt",
        batch_size = 128,
        maxlen = 100,
        test_iter = 50,
        model_type = 'DNN',
        n_id = 1000000

):
    model_path = "dnn_save_path/book_ckpt_noshuff" + model_type
    best_model_path = "dnn_best_model/book_ckpt_noshuff" + model_type

    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, batch_size, maxlen, data_in_memory=True, args=args)
        test_data = DataIterator(test_file, batch_size, maxlen, data_in_memory=True, args=args)
        whole_train_data = DataIterator(train_file, batch_size, maxlen, data_in_memory=True, args=args)
        test_emb_data = DataIterator(test_file, batch_size*10, maxlen, data_in_memory=True, max_samples=batch_size*10, args=args)


        if args.data_shuffle == 'once':
            train_data.shuffle(args.random_seed)
            whole_train_data.shuffle(args.random_seed)

        # feature_num = pkl.load(open(feature_file))
        # nid = 425972
        # nid = 426000

        BATCH_SIZE = batch_size
        SEQ_LEN = maxlen

        if model_type == 'DNN':
            model = Model_DNN(n_id, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, args=args)
        elif model_type == 'LR':
            model = Model_LR(n_id, BATCH_SIZE, SEQ_LEN, args=args)
        else:
            print ("Invalid model_type : %s", model_type)
            return

        if args.load_pretrain > 0:
            print "load pretrain model of iter {}".format(args.load_pretrain)

            # if args.optimizer != 'sgd':
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            variables_to_restore_full = tf.global_variables()
            variables_to_restore = []

            for var in variables_to_restore_full:
                # print var.name
                # print var.name
                if 'Momentum' in var.name or 'Adam' in var.name or 'RMSProp' in var.name or "Metrics" in var.name:
                    print 'skip {}'.format(var.name)

                    continue
                variables_to_restore.append(var)

            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, model_path+"--"+str(args.load_pretrain))
            # else:
            #     model.restore(sess, model_path+"--"+str(args.load_pretrain))
            print "eval pretrain model".format(args.load_pretrain)

            if args.whole_train_set:
                whole_train_auc, whole_train_loss, whole_train_acc, whole_train_aux_loss = eval(sess, whole_train_data, model, best_model_path, batch_size)
                print('                                                                                          whole_train_auc: %.4f ----whole_train_loss: %.4f ---- whole_train_accuracy: %.4f ---- whole_test_aux_loss: %.4f' % (whole_train_auc, whole_train_loss, whole_train_acc, whole_train_aux_loss))
            test_auc, test_loss, test_acc, test_aux_loss = eval(sess, test_data, model, best_model_path, batch_size, iter=0)
            print('                                                                                          test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % (test_auc, test_loss, test_acc, test_aux_loss))

        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        sys.stdout.flush()
        print('training begin')
        sys.stdout.flush()



        start_time = time.time()
        iter = 0
        lr = args.lr
        train_auc_list = []
        test_auc_list = []
        test_loss_list = []
        train_loss_list = []
        fc_grad_list_all = [] #  fc grad
        emb_grad_list_all = []
        train_loss_whole_list = []
        train_auc_whole_list = []
        grad_dict = defaultdict(list)


        for itr in range(args.epochs):
            print("epoch"+str(itr))
            if args.data_shuffle == 'every':
                train_data.shuffle(seed=itr+args.random_seed)
                whole_train_data.shuffle(seed=itr+args.random_seed)

            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            train_auc_sum = 0.0
            remain_iter = test_iter

            train_data_pool,_stop,_ = generator_queue(train_data)
            fc_grad_list = []
            emb_grad_list = []
            paras_pre = None
            while True and iter<args.iter:
                if  _stop.is_set() and train_data_pool.empty():
                    break
                if not train_data_pool.empty():
                    src,tgt = train_data_pool.get()
                else:
                    continue
                nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask = prepare_data(src, tgt)
                remain_iter -= 1

                loss, acc, aux_loss, update_dict, cal_dict = model.train(sess, [nick_id, item_id, cate_id, hist_item, hist_cate, neg_item, neg_cate, hist_mask, label, lr])
                for key in update_dict:
                    if 'grad' in key:
                        grad_dict[key].append(update_dict[key])

                train_auc_sum += 0.0
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                iter += 1
                sys.stdout.flush()
#                if iter < 2500:
#                    continue
                if remain_iter == 0:
                    print(("current iter: {} loss: {} acc: {}").format(iter, loss, acc))
                    if args.whole_train_set:

                        whole_train_auc, whole_train_loss, whole_train_acc, whole_train_aux_loss = eval(sess, whole_train_data, model, best_model_path, batch_size)
                        print('                                                                                          whole_train_auc: %.4f ----whole_train_loss: %.4f ---- whole_train_accuracy: %.4f ---- whole_test_aux_loss: %.4f' % (whole_train_auc, whole_train_loss, whole_train_acc, whole_train_aux_loss))
                        train_auc_whole_list.append(whole_train_auc)
                        train_loss_whole_list.append(whole_train_loss)
                    remain_iter = test_iter
                    print(("iter: {} loss: {} acc: {}").format(iter, loss, acc))

                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f ----  train_auc:%.5f' % \
                                          (iter, loss_sum / test_iter, accuracy_sum / test_iter,  aux_loss_sum / test_iter,  train_auc_sum/test_iter))
                    if args.corruption_percent > 0:
                        # do not test
                        test_auc, test_loss, test_acc, test_aux_loss = 0,0,0,0
                    else:
                        test_auc, test_loss, test_acc, test_aux_loss = eval(sess, test_data, model, best_model_path, batch_size)
                        print('                                                                                          test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % (test_auc, test_loss, test_acc, test_aux_loss))
                    


                    train_auc_list.append(train_auc_sum/test_iter)
                    train_loss_list.append(loss_sum/test_iter)

                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                    train_auc_sum = 0.0
                    test_time = time.time()
                    print("test interval: "+str((test_time-start_time)/60.0)+" min")
                    test_auc_list.append(test_auc)
                    test_loss_list.append(test_loss)
                # if (iter % save_iter) == 0:
                #     print('save model iter: %d' %(iter))
                    # model.save(sess, model_path+"--"+str(iter))
            fc_grad_list_all.append(fc_grad_list)
            emb_grad_list_all.append(emb_grad_list)
            if args.save_model:
                print('save model iter: %d' %(iter))
                model.save(sess, model_path+"--"+str(iter))

            print '\n\n\nepoch: ', itr
            if args.print_grad:
                for key in grad_dict:
                    grad_dict[key] = (np.around(grad_dict[key], decimals=6)).tolist()[:]
                print "grad_dict=",dict(grad_dict)

            print "\n\ntrain_auc_list=",(np.around(train_auc_whole_list, decimals=5)).tolist()
            print "test_auc_list=",(np.around(test_auc_list, decimals=5)).tolist()
            print "train_loss=",(np.around(train_loss_whole_list, decimals=5)).tolist()
            print "test_loss=",(np.around(test_loss_list, decimals=5)).tolist()



if __name__ == '__main__':
    print sys.argv
    SEED = args.random_seed
    Model_Type = args.model_type

    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    n_id = args.n_id
    if args.filter_percent > 0:
        n_id = int(n_id*args.filter_percent) +1

    if args.rehash > 0:
        n_id = int(n_id*args.rehash) +1

    train(model_type=Model_Type, n_id=n_id, batch_size=args.batch_size, test_iter=args.test_iter)
