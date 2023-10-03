#codine=utf8
import os
import sys
import pickle
import jieba.posseg as pseg

from .Define import *

class InverseIndexDict:
    def __init__(self, stop_file, user_file, sen_idx_file, word_doc_file, logger):
        self.use_pos = ["a", "ad", "an", "d", "f", "i", "j", "l", "m", "Ng", "n", "nr", "ns", "nt", "nz", "o", "s", "t", "v", "vd", "vn", "z", "un"]
        self.logger = logger
        self.stop_file = stop_file
        self.stopwords = set()
        self.user_file = user_file

        self.word_docids = dict()##word, docid
        self.docid_infos = dict()#docid, text words occurtimes labels

    def Init(self):
        if self._load_stopwords() != 0:
            self.logger.error("{} load stopwords fail!".format(self.__class__.__name__))
            return -1

        sen_ids = dict()
        sen_occu_times = dict()
        sen_labels = dict()

        if self._sta_occu(sen_ids, sen_occu_times, sen_labels) != 0:
            self.logger.error("{} load _sta_occu fail!".format(self.__class__.__name__))
            return -1

        self._create_dict(sen_occu_times, sen_ids, sen_labels)
        self.logger.debug("{} init success! docid_infos : {}".format(self.__class__.__name__, len(self.docid_infos)))
        return 0

    def _load_stopwords(self):
        if os.path.exists(self.stop_file) == False:
            return -1
        with open(self.stop_file, "r") as f:
            for line in f.readlines():
                word = line.strip()
                self.stopwords.add(word)
        return 0

    def _select_keyword(self, sentence, words):
        ori_info = pseg.cut(sentence)
        # words = jieba.cut(sentence)
        # print(sentence)
        for word, flag in ori_info:
            if word not in self.stopwords and word.strip() != "":# and flag in self.use_pos:
                words.append(word)

    def _sta_occu(self, sen_ids, sen_occu_times, sen_labels):
        if os.path.exists(self.user_file) == False:
            return -1
        with open(self.user_file, 'rb') as handle:
            texts = pickle.load(handle)
            labels = pickle.load(handle)
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
        return 0

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
            self.docid_infos[docid]["sen_new"] = "".join(sen_keywords[sen])
            self.docid_infos[docid]["keys"] = sen_keywords[sen]
            self.docid_infos[docid]["occur_time"] = sen_occu_times[sen_ids[sen]]
            self.docid_infos[docid]["label"] = sen_labels[sen]
                  
    def _sort_res(self, target_word_docids, sorted_res):
        to_idx = dict()
        to_flag = dict()
        for word_idx in target_word_docids:
            for idx in target_word_docids[word_idx]:
                if idx not in sorted_res:
                    sorted_res[idx] = 0
                sorted_res[idx] += 1
        
    def seach_cand_docs(self, target_sen, each_word_max=1000, total_max=10):
        target_words = list()
        self._select_keyword(target_sen, target_words)
        target_sen_new = "".join(target_words)

        res = dict()
        idx = 0
        for word in target_words:
            cur_cand = list()
            if word in self.word_docids:
                cur_cand = self.word_docids[word][:each_word_max]
            res.setdefault(idx, list())
            res[idx] = sorted(cur_cand)
            idx += 1
        sorted_res = {}
        self._sort_res(res, sorted_res)
        matched = sorted(sorted_res.items(), key=lambda k: k[1], reverse=True)
        matched_labels = []
        for idx, word_num in matched:
            if self.docid_infos[idx]["sen_new"] == target_sen_new:
                #print(self.docid_infos[idx]["sen_new"], target_sen_new)
                matched_labels.append(self.docid_infos[idx]["label"])
                break

            '''if word_num == 1:
                continue
            
            idx_len = len(self.docid_infos[idx]["keys"])
            key_len = len(target_words)

            if idx_len < 5 :
                if self.docid_infos[idx]["sen_new"].find(target_sen_new) > -1:
                    matched_labels.append(self.docid_infos[idx]["label"])
                    break
                else:
                    continue

            if key_len < 5:
                if target_sen_new.find(self.docid_infos[idx]["sen_new"]) > -1:
                    matched_labels.append(self.docid_infos[idx]["label"])
                    break
                else:
                    continue

            if word_num/len(self.docid_infos[idx]["keys"]) < 0.8:
               continue

            if word_num/key_len > 0.9:
                matched_labels.append(self.docid_infos[idx]["label"])
                break'''
        return matched_labels

if __name__ == "__main__":
    stop_file = "stop_words.utf8"#sys.argv[1]
    user_file = ""#sys.argv[2]
    sen_idx_file = sys.argv[3]
    word_doc_file = sys.argv[4]
    s = InverseIndexDict(stop_file, user_file, sen_idx_file, word_doc_file)
    s.main()
