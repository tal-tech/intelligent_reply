# ******************************************************
# Author       : liuqiongqiong1
# Last modified: 2019-06-24 15:41
# Email        : liuqiongqiong1@100tal.com
# Filename     : create_dict.py
# ******************************************************

#codine=utf8

import sys
import pickle
import jieba.posseg as pseg

from src.Define import *

class InverseIndexDict:
    def __init__(self, stop_file, user_file, sen_idx_file, word_doc_file):
        self.use_pos = ["a", "ad", "an", "d", "f", "i", "j", "l", "m", "Ng", "n", "nr", "ns", "nt", "nz", "o", "s", "t", "v", "vd", "vn", "z", "un"]

        self.stop_file = stop_file
        self.stopwords = set()

        self.user_file = user_file

        self.label_infos = {}

        self.word_docids = dict()##word, docid
        self.docid_infos = dict()#docid, text words occurtimes labels

        self.label_infos = {}
        self.label_name2tag = {}

    def _load_stopwords(self):
        with open(self.stop_file, "r") as f:
            for line in f.readlines():
                word = line.strip()
                self.stopwords.add(word)

    def _select_keyword(self, sentence, words):
        ori_info = pseg.cut(sentence)
        # words = jieba.cut(sentence)
        # print(sentence)
        for word, flag in ori_info:
            # if flag in self.use_pos:
            if word not in self.stopwords and word.strip() != "":# and flag in self.use_pos:
                words.append(word)

        #print(" ".join(words))

    def _sta_occu(self):
        sen_ids = dict()
        sen_occu_times = dict()
        sen_labels = dict()

        with open(self.user_file, 'rb') as handle:
            texts = pickle.load(handle)
            labels = pickle.load(handle)
            #vectors = pickle.load(handle)
        if len(texts) == 0 or len(labels) == 0:
            return -1
        if len(texts) != len(labels):
            return -1

        for i in range(len(texts)):
            cur_idx = i
            text = texts[i]
            label = labels[i]
            if text in sen_ids:
                cur_idx = sen_ids[text]
            else:
                sen_ids[text] = cur_idx
            sen_ids[text] = cur_idx 
            sen_occu_times.setdefault(cur_idx, 0)
            sen_occu_times[cur_idx] += 1
            sen_labels[text] = label
        return sen_occu_times, sen_ids, sen_labels

    def _create_dict(self, sen_occu_times, sen_ids, sen_labels):
        sen_keywords = dict()
        word_docids = dict()

        for text in sen_ids:
            cur_id = sen_ids[text]
            words = list()
            self._select_keyword(text, words)
            sen_keywords[text] = words
            for word in words:
                word_docids.setdefault(word, list())
                word_docids[word].append(cur_id)

        print("word_docids size: %d" % (len(word_docids)))
        # sort each word's doc ids by occur times
        for word in word_docids:
            docid_list = word_docids[word]
            docid_occu_times = dict()
            for doc in docid_list:
                if doc not in sen_occu_times:
                    continue
                docid_occu_times[doc] = sen_occu_times[doc]
            d = sorted(docid_occu_times.items(), key = lambda k : k[1], reverse = True)
            self.word_docids.setdefault(word, list()) 
            for k in d:
                self.word_docids[word].append(k[0])

        for sen in sen_ids:
            docid = sen_ids[sen]
            self.docid_infos.setdefault(docid, dict())
            self.docid_infos[docid]["sen"] = sen
            self.docid_infos[docid]["keys"] = sen_keywords[sen]
            self.docid_infos[docid]["occur_time"] = sen_occu_times[sen_ids[sen]]
            self.docid_infos[docid]["label"] = sen_labels[sen]
            
    def create_dict(self):
        print("start load stopwords!")
        self._load_stopwords()
        print("start sta query occus!")
        sen_occu_times, sen_ids, sen_labels = self._sta_occu()
        print("sen num: %d" % (len(sen_ids)))
        print("start create dict!")
        self._create_dict(sen_occu_times, sen_ids, sen_labels)
        print("end create dict!")


    def ParseLabelConfig(self, config_file):
        with open(config_file, 'r') as fin:
            for line in fin:
                line = line.strip()
                line = line.split("\t")
                if len(line) < 3:
                    self.logger.error("perse label line error!")
                    return -1
                label_i = LabelInfo()
                label_i.label_id = int(line[0])
                label_i.label_name = line[1]
                label_i.label_info = line[2]
                if label_i.label_id in self.label_infos:
                    return -1
                self.label_infos[label_i.label_id] = label_i
                if label_i.label_name in self.label_name2tag:
                    return -1
                self.label_name2tag[label_i.label_name] = label_i.label_id
        if len(self.label_infos) == 0 or len(self.label_infos) != len(self.label_name2tag):
            return -1
        return 0

    def _sort_res(self, target_word_docids, sorted_res):
        #print(target_word_docids)
        to_idx = dict()
        to_flag = dict()
        for word_idx in target_word_docids:
            for idx in target_word_docids[word_idx]:
                if idx not in sorted_res:
                    sorted_res[idx] = 0
                sorted_res[idx] += 1
        #print("sorted_res 0", sorted_res)
        '''sorted_res = {}
        for idx in 
        while len(to_flag) != len(target_word_docids):
            # find min doc idx
            min_idx = len(self.docid_infos)
            for word_idx in target_word_docids:
                to_idx.setdefault(word_idx, 0)
                cur_to = to_idx[word_idx]
                if cur_to < len(target_word_docids[word_idx]):
                    cur_to_idx = target_word_docids[word_idx][to_idx[word_idx]]
                    print(word_idx, cur_to_idx, min_idx)
                    if cur_to_idx < min_idx:
                        min_idx = cur_to_idx
                    else:
                        to_flag[word_idx] = True
                else:
                    to_flag[word_idx] = True

            # next
            for word_idx in target_word_docids:
                if word_idx in to_flag:
                    continue
                cur_idx = target_word_docids[word_idx][to_idx[word_idx]]
                if cur_idx == min_idx:
                    # move to next
                    to_idx[word_idx] += 1

                    sorted_res.setdefault(cur_idx, 0)
                    sorted_res[cur_idx] += 1'''

    def seach_cand_docs(self, target_sen, sorted_res, each_word_max=1000, total_max=10):
        target_words = list()
        self._select_keyword(target_sen, target_words)
        res = dict()
        idx = 0
        for word in target_words:
            cur_cand = list()
            if word in self.word_docids:
                cur_cand = self.word_docids[word][:each_word_max]
            res.setdefault(idx, list())
            res[idx] = sorted(cur_cand)
            idx += 1
        self._sort_res(res, sorted_res)
        #print("sorted_res", sorted_res)
        matched = sorted(sorted_res.items(), key=lambda k: k[1], reverse=True)
        matched = matched[:10]
        # can change
        #print("sorted_res 1", matched)
        scores_dict = dict()
        for idx, word_num in matched:
            if self.docid_infos[idx]["sen"] == target_sen:
                score = 1
            else:
                score = float(word_num) / (len(target_words))  # + len(self.docid_infos[idx]["keys"]))
            scores_dict[self.docid_infos[idx]["label"]] = [score, self.docid_infos[idx]["sen"]]
        matched = sorted(scores_dict.items(), key=lambda k: k[1], reverse=True)
        #print("scores_dict 1", matched)

        matched_labels=[]
        distances = []
        texts = []
        for label, score_text in matched:
            matched_labels.append(label)
            distances.append(score_text[0])
            texts.append(score_text[1])
        return matched_labels, distances, texts

if __name__ == "__main__":
    stop_file = "stop_words.utf8"#sys.argv[1]
    user_file = ""#sys.argv[2]
    sen_idx_file = sys.argv[3]
    word_doc_file = sys.argv[4]
    s = InverseIndexDict(stop_file, user_file, sen_idx_file, word_doc_file)
    s.main()
