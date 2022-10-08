import numpy as np
import tensorflow as tf
# from utils import *
import utils
from tensorflow.python.ops.rnn_cell import GRUCell


class Model(object):
    def __init__(self, nid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN,  Flag="DNN", args=None):
        self.model_flag = Flag
        self.reg = False
        self.args = args

        self.update_dict = {}
        self.cal_dict = {}
        self.is_training = True

        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.cate_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cate_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.cate_batch_ph = tf.placeholder(tf.int32, [None, ], name='cate_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, 1], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.embeddings_var = tf.get_variable("embedding_var", [nid, EMBEDDING_DIM], trainable=True)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.embeddings_var, self.uid_batch_ph)

            self.mid_batch_embedded = tf.nn.embedding_lookup(self.embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.embeddings_var, self.mid_his_batch_ph)

            self.cate_batch_embedded = tf.nn.embedding_lookup(self.embeddings_var, self.cate_batch_ph)
            self.cate_his_batch_embedded = tf.nn.embedding_lookup(self.embeddings_var, self.cate_his_batch_ph)



            #

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cate_batch_embedded], axis=1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded,self.cate_his_batch_embedded], axis=2) * tf.reshape(self.mask,(-1, SEQ_LEN, 1))




        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
        #mean pooling
        self.item_his_eb_sum = tf.multiply(self.item_his_eb_sum, 1.0/(0.0000001+tf.reduce_sum(self.mask, 1, keep_dims=True)))
        #



    def build_fcn_net(self, inp):
        # bn1 = utils.bn(inp, is_training=self.is_training, scale=True, scope='bn')
        bn1 = inp
        if self.args.dropout > 0 and self.args.dropout<1:
            bn1 = tf.nn.dropout(bn1, 1-self.args.dropout)
        # fc1 = utils.fc(bn1, 200, act_type='dice', is_training=self.is_training, scope='fc1')
        fc1 = utils.fc(bn1, 200, act_type=self.args.activation, is_training=self.is_training, scope='fc1')

        if self.args.dropout > 0 and self.args.dropout<1:
            fc1 = tf.nn.dropout(fc1, 1-self.args.dropout)
        # fc2 = utils.fc(fc1, 80, act_type='dice', is_training=self.is_training, scope='fc2')
        fc2 = utils.fc(fc1, 80, act_type=self.args.activation, is_training=self.is_training, scope='fc2')

        if self.args.dropout > 0 and self.args.dropout<1:
            fc2 = tf.nn.dropout(fc2, 1-self.args.dropout)
        fc3 = utils.fc(fc2, 1, act_type='', is_training=self.is_training, scope='fc3')
        self.build_loss(fc3)


    def build_fcn_net_variable(self, inp):
        neurons = [int(200*self.args.neuron) for i in range(self.args.nlayers-1)] + [int(80*self.args.neuron)]
        fc_out = inp
        for i in range(len(neurons)):
            fc_out = utils.fc(fc_out, neurons[i], act_type=self.args.activation, is_training=self.is_training, scope='fc'+str(i+1))

        fc_last = utils.fc(fc_out, 1, act_type='', is_training=self.is_training, scope='fc_last')
        self.build_loss(fc_last)


    def build_loss(self, inp):

        self.y_hat = tf.nn.sigmoid(inp) + 0.00000001
        self.y_hat = tf.clip_by_value(self.y_hat, 1e-6, 0.999999)


        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph + tf.log(1 - self.y_hat) * (1-self.target_ph))

            self.loss = ctr_loss
            # if self.use_negsample:
            #     self.loss += self.aux_loss
            # if self.reg:
            #     self.loss += self.reg_loss

            tf.summary.scalar('loss', self.loss)
            if self.args.optimizer == 'adam':
                print 'adam optimizer'
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.args.opt_decay)#.minimize(self.loss)
            elif self.args.optimizer == 'momentum':
                print 'Momentum optimizer'
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)#.minimize(self.loss)
            elif self.args.optimizer == 'rmsprop':
                print 'RMSprop optimizer'
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=self.args.opt_decay)#.minimize(self.loss)
            elif self.args.optimizer == 'adagrad':
                print 'adagrad optimizer'
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)#.minimize(self.loss)
            elif self.args.optimizer == 'adadelta':
                print 'adadelta optimizer'
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)#.minimize(self.loss)

            else:
                print 'sgd optimizer'
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)#.minimize(self.loss)


            #only train embedding
            tvars = tf.trainable_variables()
            if self.args.train_para == 'emb':
                print 'only train emb'
            # tvars = [v for v in tvars if 'embedding_var' in v.name]
                tvars = [v for v in tvars if 'embedding_var' in v.name]
            elif self.args.train_para == 'fc':
                print 'only train fc'
                tvars = [v for v in tvars if 'embedding_var' not in v.name]
            elif self.args.train_para == 'fc3':
                print 'only train fc3'
                new_tvars = []
                for v in tvars:
                    if 'fc3' in v.name:
                        print v.name
                    new_tvars.append(v)
                tvars = new_tvars
            elif self.args.train_para == 'all':
                print 'train all paras'
            else:
                print "invalid train paras!"
                ValueError
            print ("total params", np.sum([np.prod(v.get_shape().as_list()) for v in tvars]))
            grads_and_vars = self.optimizer.compute_gradients(self.loss, tvars)
            grads_and_vars_for_wd = [item for item in grads_and_vars]


            # print grads
            updated_paras = {}
            for i in range(len(grads_and_vars)):
                grad = grads_and_vars[i][0]
                name = tvars[i]
                # if grad is not None and ('kernel' in name.name or 'bias' in name.name):

                if 'emb' not in name.name and 'fc' in name.name and 'kernel' in name.name and 'dice' not in name.name:

                # if grad is not None:
                    # value = tf.Print(value, [name.name,value])
                    # grads_sum += tf.reduce_sum(tf.abs(grad))
                    # grads_sum += tf.reduce_sum(grad*grad)
                    # paras_list.append(grads_and_vars[i][1])
                    para_tmp = tf.Variable(grads_and_vars[i][1])
                    updated_paras[name.name] = [para_tmp.assign(grads_and_vars[i][1]), grads_and_vars[i][1]]
                elif 'emb' in name.name:
                    idx_unique, _ = tf.unique(grad.indices)
                    updated_paras[name.name] = [tf.identity(tf.gather(grads_and_vars[i][1], idx_unique)), grads_and_vars[i][1], idx_unique]

            # self.cal_dict['grad'] = grads_sum
            # self.cal_dict['paras'] = paras_list


            # grads_and_vars = self.optimizer.compute_gradients(self.loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                # print var, grad

                if grad is not None:
                    # grad = tf.Print(grad, [grad], summarize=200)
                    grads_and_vars[idx] = (tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), var)




            # self.optimizer = self.optimizer.apply_gradients(grads_and_vars)

            with tf.control_dependencies([item[0] for item in updated_paras.values()]): #bak
                if self.args.weight_decay >0:
                    l2_loss = 0.0

                    for idx, (grad, var) in enumerate(grads_and_vars_for_wd):
                        if "embedding_var" in var.name:
                            # print tf.gather(var, grad.indices)
                            emb_para = tf.gather(var, updated_paras[var.name][2])
                            # emb_para = tf.Print(emb_para, [grad.indices], summarize=10000)
                            # l2_loss += tf.reduce_sum(tf.square(emb_para))
                            l2_loss += tf.reduce_sum(emb_para*emb_para)

                            pass
                        else:
                            l2_loss += tf.reduce_sum(var*var)
                            pass
                    # print "l2_loss", l2_loss
                    # l2_loss = tf.Print(l2_loss, ["loss:", self.loss, "  l2_loss:", l2_loss], summarize=200)
                    sgd = tf.train.GradientDescentOptimizer(learning_rate=self.args.weight_decay)
                    decay_op = sgd.minimize(l2_loss)
                else:
                    decay_op = tf.Variable([0])

            with tf.control_dependencies([decay_op]+[item[0] for item in updated_paras.values()]):
                self.optimizer = self.optimizer.apply_gradients(grads_and_vars)

            # calculate paras change
            with tf.control_dependencies([self.optimizer]):
                # cal emb_para grad
                for idx, (grad, var) in enumerate(grads_and_vars_for_wd):
                    if 'emb' in var.name:
                        paras_grad = updated_paras[var.name][0] - tf.gather(var, updated_paras[var.name][2])
                    elif var.name in updated_paras:
                        paras_grad = updated_paras[var.name][0] - var

                # self.update_dict['emb_grad'] = tf.reduce_mean(emb_grad*emb_grad)
                    self.update_dict[var.name+'_grad'] = tf.reduce_mean(tf.abs(paras_grad))
                    self.update_dict[var.name+'_grad_max'] = tf.reduce_max(tf.abs(paras_grad))

                # weight decay

                #


            ####################


            # Accuracy metric

            ## debug start ##
            # y_hat = tf.Print(self.y_hat, ['y_hat', tf.squeeze(self.y_hat)], summarize=100)
            # target_ph = tf.Print(self.target_ph, ['target_ph', tf.squeeze(self.target_ph)],summarize=100)
            # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y_hat), target_ph), tf.float32))
            ## debug end ###

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()



    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask = None, stag = None):
        #mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]

        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask

        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 1, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        #y_hat = tf.nn.softmax(dnn3) + 0.000001
        y_hat = tf.nn.sigmoid(dnn3) + 0.000001
        return y_hat

    def init_uid_weight(self, sess, uid_weight):
        sess.run(self.uid_embedding_init,feed_dict={self.uid_embedding_placeholder: uid_weight})

    def init_mid_weight(self, sess, mid_weight):
        sess.run([self.mid_embedding_init],feed_dict={self.mid_embedding_placeholder: mid_weight})

    def save_mid_embedding_weight(self, sess):
        embedding = sess.run(self.mid_embeddings_var)
        return embedding

    def save_uid_embedding_weight(self, sess):
        embedding = sess.run(self.uid_bp_memory)
        return embedding

    def train(self, sess, inps):
        self.is_training = True

        loss, accuracy, _,update_dict,  cal_dict = sess.run([self.loss, self.accuracy, self.optimizer, self.update_dict, self.cal_dict], feed_dict={
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.cate_batch_ph: inps[2],
            self.mid_his_batch_ph: inps[3],
            self.cate_his_batch_ph: inps[4],
            self.mask: inps[7],
            self.target_ph: inps[8],
            self.lr: inps[9]
        })
        aux_loss = 0
        return loss, accuracy, aux_loss, update_dict, cal_dict

    def calculate(self, sess, inps):
        self.is_training = False

        probs, loss, accuracy, cal_dict = sess.run([self.y_hat, self.loss, self.accuracy, self.cal_dict], feed_dict={
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.cate_batch_ph: inps[2],
            self.mid_his_batch_ph: inps[3],
            self.cate_his_batch_ph: inps[4],
            self.mask: inps[7],
            self.target_ph: inps[8]
        })
        aux_loss = 0
        return probs, loss, accuracy, aux_loss, cal_dict

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

class Model_DNN(Model):
    def __init__(self,nid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256, args=None):
        super(Model_DNN, self).__init__(nid, EMBEDDING_DIM, HIDDEN_SIZE,
                                           BATCH_SIZE, SEQ_LEN, Flag="DNN", args=args)
        self.args = args
        if self.args.dataset == 'taobao':
            print "taobao dataset, do not use user ID"
            # do not use user id, because one user has only one sample, so user id does not provide any information
            inp = tf.concat([self.item_eb, self.item_his_eb_sum], 1)
        else:
            inp = tf.concat([self.item_eb, self.item_his_eb_sum, self.uid_batch_embedded], 1)

        self.cal_dict['inp'] = inp

        if self.args.neuron == 1 and self.args.nlayers == 2:
            self.build_fcn_net(inp)
        else:
            self.build_fcn_net_variable(inp)


class Model_LR(Model):
    def __init__(self,nid, BATCH_SIZE, SEQ_LEN=256, args=None):
        super(Model_LR, self).__init__(nid, EMBEDDING_DIM=1, HIDDEN_SIZE=0,
                                           BATCH_SIZE=BATCH_SIZE, SEQ_LEN=SEQ_LEN, Flag="LR", args=args)

        # inp = tf.concat([self.item_eb, self.item_his_eb_sum, self.uid_batch_embedded], 1)
        inp = tf.concat([self.item_eb, self.item_his_eb_sum, self.uid_batch_embedded], 1)


        self.cal_dict['inp'] = inp

        # inp = tf.Print(inp, ["\ninp", tf.shape(inp)], summarize=200)

        with tf.variable_scope('LR_bias', reuse=tf.AUTO_REUSE):
            bias = tf.get_variable("output_bias", shape=(1,), initializer=tf.initializers.constant(0.0))
            logit = tf.reduce_sum(inp, 1, keep_dims=True) + bias


        # logit = self.build_fcn_net_test(inp, use_dice=False)
        # logit = self.build_fcn_net_heavy(inp, use_dice=False)
        # logit = tf.Print(logit, ["\nlogit", tf.shape(logit)], summarize=200)

        self.build_loss(logit, L2=False, logit_dim=1)

