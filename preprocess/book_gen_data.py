import cPickle

files = ["train", "test"]
all_id_voc = dict(cPickle.load(open("./data/book_data/all_id_voc_book.pkl", "r")))
print "all_id_number:",len(all_id_voc)

for file in files:
    print file
    f_train = open("./data/book_data/book_{}0.txt".format(file), "r")
    f_train_map = open("./data/book_data/book_{}.txt".format(file), "w")


    all_id_voc[0] = 0

    res = ""
    for line in f_train:
        ss = line.strip("\n").split("\t")

        uid = int(ss[0])
        item_id = int(ss[1])
        cate_id = int(ss[2])
        label = int(ss[3])

        hist_item = map(int, ss[4].split(","))
        hist_cate = map(int, ss[5].split(","))

        neg_item = map(int, ss[6].split(","))
        neg_cate = map(int, ss[7].split(","))

        uid = all_id_voc[uid]
        item_id = all_id_voc[item_id]
        cate_id = all_id_voc[cate_id]

        hist_item = map(lambda x: str(all_id_voc[x]), hist_item)
        hist_cate = map(lambda x: str(all_id_voc[x]), hist_cate)
        # neg_item = map(lambda x: str(all_id_voc[x]), neg_item)
        # neg_cate = map(lambda x: str(all_id_voc[x]), neg_cate)

        hist_item = ",".join(list(hist_item))
        hist_cate = ",".join(list(hist_cate))
        # neg_item = ",".join(list(neg_item))
        # neg_cate = ",".join(list(neg_cate))
        neg_item = '0'
        neg_cate = '0'


        cur_res = str(uid)+'\t'+str(item_id)+'\t'+str(cate_id)+'\t'+str(label)+'\t'+hist_item+'\t'+hist_cate+'\t'+neg_item+'\t'+neg_cate+'\n'
        res += cur_res
    print(cur_res)
    f_train_map.write(res)


# ('n user:', 987648, '  n item:', 4039879, 'n cate:', 9411)
