import numpy as np
from collections import defaultdict


# file_list = ["./data/book_data/book_train.txt","./data/book_data/book_test.txt","./data/taobao_data/taobao_train.txt", "./data/taobao_data/taobao_test.txt"]

def add_to_dict(item, data_dict):
    if item in data_dict:
        data_dict[item] += 1
    else:
        data_dict[item] = 1
    return data_dict

def entropy(c):
    result=-1;
    if(len(c)>0):
        result=0;
    c_sum = float(np.sum(c))
    print(c_sum)
    for x in c:
        result+=(-x/c_sum)*np.log2(x/c_sum)
    return result

def read_cal(fil, his_sperate=True):
    f_r = open(fil, "r")
    print(fil, "his_sperate",his_sperate, 'reading data...')

    n_lines = 0
    for line in f_r:
        n_lines += 1
        if n_lines % 100000 == 0:
            print('line number:',n_lines)
        # if n_lines > 10000:
        #     break
        ss = line.strip("\n").split("\t")

        # print ss
        # uid = int(ss[0])
        # item_id = int(ss[1])
        # cate_id = int(ss[2])
        # label = int(ss[3])
        #

        # ids

        ids_list = [0,1,2,3]
        for idx in ids_list:
            if ss[idx] != '0':
                dict_list[idx] = add_to_dict(item=ss[idx], data_dict=dict_list[idx])
                dict_list[6] = add_to_dict(item=ss[idx], data_dict=dict_list[6])

        # hist_item = map(int, ss[4].split(","))
        # hist_cate = map(int, ss[5].split(","))
        #
        # neg_item = map(int, ss[6].split(","))
        # neg_cate = map(int, ss[7].split(","))
        #
        #
        # source.append([uid, item_id, cate_id])
        # target.append([label, 1-label])
        # hist_item_list.append(hist_item[-self.maxlen:])
        # hist_cate_list.append(hist_cate[-self.maxlen:])
        #
        # neg_item_list.append(neg_item[-self.maxlen:])
        # neg_cate_list.append(neg_cate[-self.maxlen:])
        #only hist_item and hist_cate

        if his_sperate:
            hist_item = ss[4].split(",")
            for item in hist_item:
                if item != '0':
                    dict_list[4] = add_to_dict(item=item, data_dict=dict_list[4]) #to item id
                    dict_list[6] = add_to_dict(item=item, data_dict=dict_list[6])
            hist_cate = ss[5].split(",")
            # print hist_cate

            for item in hist_cate:
                if item != '0':
                    dict_list[5] = add_to_dict(item=item, data_dict=dict_list[5]) #to item cate
                    dict_list[6] = add_to_dict(item=item, data_dict=dict_list[6])

        else:
            # # sequence
            hist_item = ss[4].split(",")
            for item in hist_item:
                if item != '0':
                    dict_list[1] = add_to_dict(item=item, data_dict=dict_list[1]) #to item id
                    dict_list[6] = add_to_dict(item=item, data_dict=dict_list[6])

            #
            hist_cate = ss[5].split(",")
            # print hist_cate

            for item in hist_cate:
                if item != '0':
                    dict_list[2] = add_to_dict(item=item, data_dict=dict_list[2]) #to item cate
                    dict_list[6] = add_to_dict(item=item, data_dict=dict_list[6])


def cal_pdf():

    idx2name = {0:'uid',1:'item id', 2:'item cate', 3:'label',4:'hist_item',5:'hist_cate',6:'all'}


    # print(dict_list)
    pdf_dict = defaultdict(list)
    tmp_sorted = [(dict_list[1][key], key) for key in dict_list[1] ]
    tmp_sorted.sort(reverse=True)
    print ("most frequent ids:", tmp_sorted[0:10])

    for i in range(len(dict_list)):
        print('\n',idx2name[i] if i in idx2name else 'unknown', '  id numbers:', len(dict_list[i]), '  entropy:', entropy(list(dict_list[i].values())),  '  mean:', np.mean(list(dict_list[i].values())), ' std:', np.std(list(dict_list[i].values())))

        cnt_list = sorted(list(dict_list[i].values()), reverse=True)
        n_bins = min(len(cnt_list)+1, 200)
        step = len(cnt_list) // n_bins + 1
        name = idx2name[i] if i in idx2name else 'unknown'
        cnt_sum = float(np.sum(cnt_list)) + 1e-8
        for j in range(n_bins):
            cnt_tmp = 0
            for k in range(step):
                if j*step+k >= len(cnt_list):
                    break
                cnt_tmp += cnt_list[j*step+k]
            if cnt_tmp > 0:
                pdf_dict[name+'_'+str(i)].append(cnt_tmp/cnt_sum)

    print "\npdf_dict=",dict(pdf_dict)


file_list = ["./data/book_data/book_train.txt","./data/book_data/book_test.txt"]
dict_list = [{} for i in range(7)]

for file in file_list:
    read_cal(file, True)
    # read_cal(file, False)
cal_pdf()




