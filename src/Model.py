import time
import pickle
import jieba
import gensim
import traceback
from configparser import ConfigParser
from collections import Counter

import torch
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from bert4keras.models import build_transformer_model
from keras.preprocessing.sequence import pad_sequences
#from bert4keras.snippets import sequence_padding
#from elmoformanylangs import Embedder

from .InverseIndexDict import *
from .DataBase import *
from .Matcher import *
from .Define import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model():
    def __init__(self, config_file, logger):
        self.config_file = config_file
        self.logger = logger

        cp = ConfigParser()
        cp.read(self.config_file)

        self.maxtext = int(cp.get("match", "max_text"))
        self.topk = int(cp.get("match", "k"))
        self.maxlen = int(cp.get("match", "max_len"))
        self.threshold = float(cp.get("match", "threshold"))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name= cp.get("model", "model")

        self.label_infos = {}
        self.GetFullMatcher(cp)
        self.GetSelect(cp)
        self.GetModel(cp)
    
    def Init(self):
        if self.InverseIndexDict.Init() != 0:
            self.logger.error("{} InverseIndexDict Init fail!".format(self.__class__.__name__))
            return -1

        if self.InitSelect() != 0:
            self.logger.error("{} InitSelect Init fail!".format(self.__class__.__name__))
            return -1

        if self.InitModel() != 0:
            self.logger.error("{} InitSelect Init fail!".format(self.__class__.__name__))
            return -1
        self.logger.debug("{} init success!".format(self.__class__.__name__))
        return 0

    def GetFullMatcher(self, config_parser):
        self.stop_file = config_parser.get("full_match", "stop_file")
        self.doc_file = config_parser.get("select", "info")
        self.sen_idx_file = config_parser.get("full_match", "sen_idx_file")
        self.word_doc_file = config_parser.get("full_match", "word_doc_file")
        self.InverseIndexDict = InverseIndexDict(self.stop_file, self.doc_file, self.sen_idx_file, self.word_doc_file, self.logger)
        
    def GetSelect(self, config_parser):
        self.database = DataBase(config_parser, self.logger)
        self.select_type = config_parser.get('select', 'model')
        self.select_path = config_parser.get("select", "path")

    def InitSelect(self):
        if self.database.Init() != 0:
            self.logger.error("{} database Init fail!".format(self.__class__.__name__))
            return -1

        self.Matcher = Matcher(self.topk, self.database.vectors, self.logger)
        if self.Matcher.Init() != 0:
            self.logger.error("{} matcher Init fail!".format(self.__class__.__name__))
            return -1

        self.label_infos = self.database.label_infos
        if os.path.exists(self.select_path) == False:
            self.logger.error("{} have no path Init fail!".format(self.__class__.__name__))
            return -1
        print(self.select_type)
        if self.select_type == "word2vector":
            self.emb = gensim.models.KeyedVectors.load_word2vec_format(self.select_path, binary=True)
            print(len(self.emb.vectors))
        else:
            self.select_model = BertModel.from_pretrained(self.select_path)
            self.select_tokenizer = BertTokenizer.from_pretrained(self.select_path)
            self.select_model = self.select_model.to(self.device)
        return 0

    def GetModel(self, config_parser):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = int(config_parser.get('model', 'num_labels'))
        self.model_path = config_parser.get('model', 'path')
    
    def InitModel(self):
        if os.path.exists(self.model_path) == False:
            self.logger.error("{} have no path Init fail!".format(self.__class__.__name__))
            return -1
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, do_lower_case=False)
        if self.num_labels == 1:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=1)
        elif self.num_labels == 2:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        return 0

    def Match(self, text, slots):
        if self.FullMatch(text, slots) > 0:
            return 0
        if len(text) > self.maxtext:
            return 0
        matched_labels, distances, matched_texts, matched_simi = self.FaissMatch(text, slots)
        if self.num_labels == 1:
            matched_score, matched_score_index_sort  = self.ModelMatch_reg(text, matched_labels, matched_simi, matched_texts, slots)
        elif self.num_labels == 2:
            matched_score, matched_score_index_sort  = self.ModelMatch_cls(text, matched_labels, matched_simi, matched_texts, slots)
        else:
            matched_score, matched_score_index_sort  = self.ModelMatch_no(text, matched_labels, matched_simi, matched_texts, slots)
        return self.GetMatchOut(text, matched_score, matched_labels, matched_texts, matched_score_index_sort, slots)
    
    def FullMatch(self, text, slots):
        matched_labels = self.InverseIndexDict.seach_cand_docs(text)
        for label in matched_labels:
            slot_i = LabelInfo()
            slot_i.label_score = 1
            slot_i.label_id = label
            slot_i.label_name = self.label_infos[label].label_name
            slot_i.label_info = self.label_infos[label].label_info
            slots.append(slot_i)
            return 1
        return 0
    
    def RegMatch(self, text, slots):
        for label in self.label_infos:
            for pattern in self.label_infos[label].label_reg:
                if not re.search(pattern, text) is None:
                    slot_i = LabelInfo()
                    slot_i.label_score = 1
                    slot_i.label_id = label
                    slot_i.label_name = self.label_infos[label].label_name
                    slot_i.label_info = self.label_infos[label].label_info
                    slots.append(slot_i)
                    return 1
        return 0

    def FaissMatch(self, text, slots):
        def cos_sim(vector_a, vector_b):
            vector_a = np.mat(vector_a)
            vector_b = np.mat(vector_b)
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            sim = cos
            return sim

        start = time.time()
        if self.select_type == "word2vector":
            words = jieba.lcut(text)[:128]
            emb_list = []
            for word in words:
                if word in self.emb.vocab:
                    emb_list.append(self.emb.get_vector(word))
                else:
                    for char_i in word:
                        if char_i in self.emb.vocab:
                            emb_list.append(self.emb.get_vector(char_i))
                    else:
                        continue
            vector_in = np.array(emb_list).mean(axis=0).reshape((1,-1))
        else:
            input_id = self.select_tokenizer.encode(text, add_special_tokens=True, max_length=self.maxlen)
            input_ids = pad_sequences([input_id], maxlen=self.maxlen, dtype="long", value=self.tokenizer.pad_token_id,
                                    truncating="post", padding="post")
            attention_masks = []
            for sent in input_ids:
                att_mask = [int(token_id != self.tokenizer.pad_token_id) for token_id in sent]
                attention_masks.append(att_mask)

            inputs = torch.tensor(input_ids)
            att_mask = torch.tensor(attention_masks)

            inputs = inputs.to(self.device)
            att_mask = att_mask.to(self.device)
            outputs = self.select_model(inputs, attention_mask=att_mask)

            vector_in = outputs[1].cpu().detach().numpy()
        matched_idxes, distances = self.Matcher.search(vector_in)
        matched_labels = []
        matched_texts = []
        matched_simi = []
        for index_i in matched_idxes:
            matched_labels.append(self.database.labels[index_i])
            matched_texts.append(self.database.texts[index_i])
            vector_i = self.database.vectors[index_i].reshape((1,-1))
            simi = cos_sim(vector_in, vector_i)
            matched_simi.append(simi)
        end = time.time()
        return matched_labels, distances, matched_texts, matched_simi
        
    def ModelMatch_cls(self, text, matched_labels, matched_simi, matched_texts, slots):
        start = time.time()
        matched_score = []
        matched_score_index_sort = []

        input_ids = []
        for i, matched_text_i in enumerate(matched_texts):
            new_text = text + "[SEP]" + matched_text_i
            input_id = self.tokenizer.encode(new_text, add_special_tokens=True, max_length=self.maxlen)
            input_ids.append(input_id)
        input_ids = pad_sequences(input_ids, maxlen=self.maxlen, dtype="long", value=self.tokenizer.pad_token_id,
                                truncating="post", padding="post")
        attention_masks = []
        for sent in input_ids:
            att_mask = [int(token_id != self.tokenizer.pad_token_id) for token_id in sent]
            attention_masks.append(att_mask)
        inputs = torch.tensor(input_ids)
        att_mask = torch.tensor(attention_masks)

        inputs = inputs.to(self.device)
        att_mask = att_mask.to(self.device)
        outputs = self.model(inputs, attention_mask=att_mask)
        outputs = outputs[0].cpu().detach().numpy()
        outputs = np.argmax(outputs, axis=1)

        for i, output_i in enumerate(outputs):
            if output_i == 1:
                matched_score_index_sort.append(i)
        return matched_simi, matched_score_index_sort

    def ModelMatch_reg(self, text, matched_labels, matched_simi, matched_texts, slots):
        start = time.time()
        '''matched_score = []
        for matched_text_i in matched_texts:
            new_text = text + "[SEP]" + matched_text_i
            input_id = self.tokenizer.encode(new_text, add_special_tokens=True, max_length=self.maxlen)
            input_ids = pad_sequences([input_id], maxlen=self.maxlen, dtype="long", value=self.tokenizer.pad_token_id,
                                    truncating="post", padding="post")
            attention_masks = []
            for sent in input_ids:
                att_mask = [int(token_id != self.tokenizer.pad_token_id) for token_id in sent]
                attention_masks.append(att_mask)

            inputs = torch.tensor(input_ids)
            att_mask = torch.tensor(attention_masks)
            outputs = self.model(inputs, attention_mask=att_mask)
            matched_score.append(outputs)'''
        
        input_ids = []
        for i, matched_text_i in enumerate(matched_texts):
            new_text = text + "[SEP]" + matched_text_i
            input_id = self.tokenizer.encode(new_text, add_special_tokens=True, max_length=self.maxlen)
            input_ids.append(input_id)
        input_ids = pad_sequences(input_ids, maxlen=self.maxlen, dtype="long", value=self.tokenizer.pad_token_id,
                                truncating="post", padding="post")
        attention_masks = []
        for sent in input_ids:
            att_mask = [int(token_id != self.tokenizer.pad_token_id) for token_id in sent]
            attention_masks.append(att_mask)

        inputs = torch.tensor(input_ids)
        att_mask = torch.tensor(attention_masks)

        inputs = inputs.to(self.device)
        att_mask = att_mask.to(self.device)
        outputs = self.model(inputs, attention_mask=att_mask)
        matched_score = outputs[0].cpu().detach().numpy().reshape(1,-1).tolist()[0]
        #print("matched_score", matched_score)
        matched_score_indexes = np.argsort(-(np.array(matched_score)))
        #print("matched_score_indexes", matched_score_indexes)
        matched_score_index_sort = []
        for i in matched_score_indexes:
            if matched_score[i] > self.threshold:
                matched_score_index_sort.append(i)
        end = time.time()
        print("reg match", end-start)
        return matched_score, matched_score_index_sort
        
        '''    elif self.model_name == "elmo":
                words = jieba.lcut(text)[:128]
                sents = [words]
                outputs = self.model.sents2elmo(sents)
                vector_in = np.array(np.mean(outputs[0],axis=0)).reshape((1,-1))
                #print(vector_in.shape)
            elif self.model_name == "bert":
                input_id = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.maxlen)
                input_ids = pad_sequences([input_id], maxlen=self.maxlen, dtype="long", value=self.tokenizer.pad_token_id,
                                        truncating="post", padding="post")
                attention_masks = []
                for sent in input_ids:
                    att_mask = [int(token_id != self.tokenizer.pad_token_id) for token_id in sent]
                    attention_masks.append(att_mask)

                inputs = torch.tensor(input_ids)
                att_mask = torch.tensor(attention_masks)
                outputs = self.model(inputs, attention_mask=att_mask)

                vector_in = outputs[1].detach().numpy()'''
                #print(vector_in)

            
        '''elif self.model_type == "Inverted index":
            sorted_res = dict()
            matched_labels, distances, matched_texts = self.InverseIndexDict.seach_cand_docs(text, sorted_res)
            label_1 = self.GetMatchedScore(matched_labels, distances, matched_texts, slots)
            return_res = label_1
        else:
            return_res = ""'''

    def ModelMatch_no(self, text, matched_labels, matched_simi, matched_texts, slots):
        matched_score_index_sort = []
        for i, matched_text_i in enumerate(matched_texts):
            matched_score_index_sort.append(i)
        return matched_simi, matched_score_index_sort
    
    def GetMatchOut1(self, text, matched_score, matched_labels, matched_texts, matched_score_index_sort, slots):
        ##直接 match score 
        labels = set()
        for idx in matched_score_index_sort:
            slot_i = LabelInfo()
            slot_i.label_score = matched_score[idx]
            slot_i.label_id = matched_labels[idx]
            slot_i.label_name = self.label_infos[slot_i.label_id].label_name
            slot_i.label_info = self.label_infos[slot_i.label_id].label_info
            print("GetMatchOut", idx, slot_i.label_name, slot_i.label_id, slot_i.label_score, self.threshold, labels)
            if slot_i.label_score > self.threshold and slot_i.label_id not in labels:
                slots.append(slot_i)
                print("add")
            labels.add(slot_i.label_id)
        print(len(slots)) 
        return slots[0]

    def GetMatchOut2(self, text, matched_score, matched_labels, matched_texts, matched_score_index_sort, slots):
        ##计算平均值
        all_nums = len(matched_labels)
        label_dicts = {}
        for idx in matched_score_index_sort:
            if matched_score[idx] < self.threshold:
                continue
            label_i = matched_labels[idx]
            if label_i not in label_dicts:
                label_dicts[label_i] = []
            label_dicts[label_i].append(matched_score[idx])

        label_score = {}
        for label_i in label_dicts:
            label_score[label_i] = np.mean(label_dicts[label_i])

        matched = sorted(label_dicts.items(), key=lambda k: k[1], reverse=True)
        
        for label_i, score_i in matched:
            slot_i = LabelInfo()
            slot_i.label_score = score_i
            slot_i.label_id = label_i
            slot_i.label_name = self.label_infos[slot_i.label_id].label_name
            slot_i.label_info = self.label_infos[slot_i.label_id].label_info
        return slots[0]
        
    def GetMatchOut(self, text, matched_score, matched_labels, matched_texts, matched_score_index_sort, slots):
        ##计算比率
        start = time.time()
        all_nums = len(matched_labels)
        label_dicts = {}
        label_scores = {}
        for idx in matched_score_index_sort:
            if matched_score[idx] < self.threshold:
                continue
            label_i = matched_labels[idx]
            if label_i not in label_dicts:
                label_dicts[label_i] = 0
            label_dicts[label_i] += 1
            if label_i not in label_scores:
                label_scores[label_i] = 0
            if matched_score[idx] > label_scores[label_i]:
                label_scores[label_i] = matched_score[idx]

        matched = sorted(label_dicts.items(), key=lambda k: k[1], reverse=True)
        for label_i, num in matched:
            slot_i = LabelInfo()
            slot_i.label_score = label_scores[label_i]
            slot_i.label_id = label_i
            slot_i.label_name = self.label_infos[slot_i.label_id].label_name
            slot_i.label_info = self.label_infos[slot_i.label_id].label_info
            if len(self.label_infos[slot_i.label_id].label_key) > 0:
                flag = False
                for key_i in self.label_infos[slot_i.label_id].label_key:
                    if text.find(key_i) > -1:
                        flag = True
                        break
                if flag == False:
                    continue
            if len(self.label_infos[slot_i.label_id].label_neg_key) > 0:
                flag = True
                for key_i in self.label_infos[slot_i.label_id].label_neg_key:
                    if text.find(key_i) > -1:
                        flag = False
                        break
                if flag == False:
                    #print("neg match", text, key_i)
                    continue
            slots.append(slot_i)
        end = time.time()
        return 0

    def GetMatchedScore(self, matched_labels, distances, matched_texts, slots):
        ## topk  计算比率
        all_nums = len(matched_labels)
        label_dicts = {}
        for i in range(all_nums):
            label_i = matched_labels[i]
            if label_i not in label_dicts:
                label_dicts[label_i] = []
            label_dicts[label_i].append(distances[i])
            #print(label_i, distances[i], matched_texts[i])

        label_top_k = 3
        label_tops = Counter(matched_labels).most_common(label_top_k)

        #print("label_dicts", label_dicts)
        #print("label_tops", label_tops)
        label_1_s = []
        max_score= 0
        for label_tops_i, num_i in label_tops:
            slot_i = LabelInfo()
            slot_i.label_score = num_i/all_nums
            slot_i.label_id = label_tops_i
            slot_i.label_name = self.label_infos[label_tops_i].label_name
            slot_i.label_info = self.label_infos[label_tops_i].label_info
            slots.append(slot_i)
            if slot_i.label_score > max_score:
                max_score = slot_i.label_score
                label_1 = slot_i.label_id
            label_1_s.append(slot_i.label_id)
        return label_1_s

    def GetMatchedScore2(self, matched_labels, distances, matched_texts, slots):
        ## topk  计算distance
        all_nums = len(matched_labels)
        label_dicts = {}
        for i in range(all_nums):
            label_i = matched_labels[i]
            if label_i not in label_dicts:
                label_dicts[label_i] = []
            label_dicts[label_i].append(distances[i])
            #print(label_i, distances[i], matched_texts[i])

        label_dis = {}
        for label_i in label_dicts:
            label_dis[label_i] = np.sum(label_dicts[label_i]) / len(label_dicts[label_i])

        label_dis = sorted(label_dis.items(), key=lambda k: k[1], reverse=True)


        label_top_k = 3
        label_tops = label_dis[:label_top_k]

        #print("label_dicts", label_dicts)
        #print("label_tops", label_tops)
        label_1_s = []
        max_score= 0
        for label_tops_i, dis_i in label_tops:
            slot_i = LabelInfo()
            slot_i.label_score = dis_i
            slot_i.label_id = label_tops_i
            slot_i.label_name = self.label_infos[label_tops_i].label_name
            slot_i.label_info = self.label_infos[label_tops_i].label_info
            slots.append(slot_i)
            if slot_i.label_score > max_score:
                max_score = slot_i.label_score
                label_1 = slot_i.label_id
            label_1_s.append(slot_i.label_id)
        return label_1_s

    def GetMatchedScore3(self, matched_labels, distances, matched_texts, slots):
        ## topk  计算distance*比率
        all_nums = len(matched_labels)
        label_dicts = {}
        for i in range(all_nums):
            label_i = matched_labels[i]
            if label_i not in label_dicts:
                label_dicts[label_i] = []
            label_dicts[label_i].append(distances[i])
            #print(label_i, distances[i], matched_texts[i])

        label_dis = {}
        for label_i in label_dicts:
            label_dis[label_i] = np.sum(label_dicts[label_i])

        label_dis = sorted(label_dis.items(), key=lambda k: k[1], reverse=True)


        label_top_k = 3
        label_tops = label_dis[:label_top_k]

        #print("label_dicts", label_dicts)
        #print("label_tops", label_tops)
        label_1_s = []
        max_score= 0
        for label_tops_i, dis_i in label_tops[:1]:
            slot_i = LabelInfo()
            slot_i.label_score = dis_i
            slot_i.label_id = label_tops_i
            slot_i.label_name = self.label_infos[label_tops_i].label_name
            slot_i.label_info = self.label_infos[label_tops_i].label_info
            slots.append(slot_i)
            if slot_i.label_score > max_score:
                max_score = slot_i.label_score
                label_1 = slot_i.label_id
            label_1_s.append(slot_i.label_id)
        return label_1_s

    

