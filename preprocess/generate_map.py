import cPickle
from collections import defaultdict
file_list = {"./data/book_data/book_train0.txt", "./data/book_data/book_test0.txt"}
all_id_dict = defaultdict(int)
mid_dict = defaultdict(int)
cate_dict = defaultdict(int)
uid_dict = defaultdict(int)

iddd = 0
for fil in file_list:
    f_r = open(fil, "r")
    for line in f_r:
        ss = line.strip("\n").split("\t")

        uid = int(ss[0])
        item_id = int(ss[1])
        cate_id = int(ss[2])
        label = int(ss[3])

        hist_item = map(int, ss[4].split(","))
        hist_cate = map(int, ss[5].split(","))

        neg_item = map(int, ss[6].split(","))
        neg_cate = map(int, ss[7].split(","))

        # if uid in cate_dict or uid in mid_dict:
        #     print('ERROR, uid', uid)
        uid_dict[uid] += 1
        all_id_dict[uid] += 1

        # if item_id in cate_dict or item_id in uid_dict:
        #     print('ERROR, uid', item_id)
        mid_dict[item_id] += 1
        all_id_dict[item_id] += 1

        # if cate_id in uid_dict or cate_id in mid_dict:
        #     print('ERROR, cate', cate_id)

        cate_dict[cate_id] += 1
        all_id_dict[cate_id] += 1

        for each_item in hist_item:
            if each_item > 0:
                # if each_item in cate_dict or each_item in uid_dict:
                #     print('ERROR, item id', each_item)
                mid_dict[each_item] += 1
                all_id_dict[each_item] += 1

        for each_cate in hist_cate:
            if each_cate > 0:
                # if each_cate in mid_dict or each_cate in uid_dict:
                #     print('ERROR, cate', each_cate)
                cate_dict[each_cate] += 1
                all_id_dict[each_cate] += 1

        ####### do not use neg######
        # for each_item in neg_item:
        #     if each_item > 0:
        #         if each_item in cate_dict or each_item in uid_dict:
        #             print('ERROR, item id', each_item)
        #         mid_dict[each_item] += 1
        #         all_id_dict[each_item] += 1
        #
        # for each_cate in neg_cate:
        #     if each_cate > 0:
        #         if each_cate in mid_dict or each_cate in uid_dict:
        #             print('ERROR, cate', each_cate)
        #         cate_dict[each_cate] += 1
        #         all_id_dict[each_cate] += 1


sorted_all_id_dict = sorted(dict(all_id_dict).iteritems(), key=lambda x:x[1], reverse=True)
print('n user:', len(uid_dict), '  n item:', len(mid_dict), 'n cate:', len(cate_dict))
print("all_id number:", len(sorted_all_id_dict))



all_id_voc = {}
all_id_voc["default_id"] = 1
all_id_voc[""] = 1
index = 2
for key, value in sorted_all_id_dict:
    all_id_voc[key] = index
    index += 1
print("all_id number:", index)

cPickle.dump(all_id_voc, open("./data/book_data/all_id_voc_book.pkl", "w"))
# cPickle.dump(all_id_voc, open("yueyi_all_id_voc_book_no_neg.pkl", "w"))

