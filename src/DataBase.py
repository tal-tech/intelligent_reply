import os
import pickle
import traceback
from configparser import ConfigParser

from src.Define import *


class DataBase():
    def __init__(self, config_parser, logger):
        self.config_parser = config_parser
        self.logger = logger

        self.label_infos = {}
        self.label_name2tag = {}
        self.texts = []
        self.labels = []
        self.vectors = []

    def Init(self):
        if self.ParseLabelConfig(self.config_parser.get("database", "label")) != 0:
            self.logger.error("{} label init fail!".format(self.__class__.__name__))
            return -1

        if self.ParseInfoConfig(self.config_parser.get("select", "info")) != 0:
            self.logger.error("{} label info fail!".format(self.__class__.__name__))
            return -1
        self.logger.debug("{} init success!".format(self.__class__.__name__))
        return 0

    def ParseLabelConfig(self, config_file):
        if os.path.exists(config_file) == False:
            return -1
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
                if len(line) > 3:
                    label_i.label_key = line[3].split("|")
                if len(line) > 4:
                    label_i.label_neg_key = line[4].split("|")
                    #print("neg_label", label_i.label_neg_key, line)
                if label_i.label_id in self.label_infos:
                    return -1
                self.label_infos[label_i.label_id] = label_i
                if label_i.label_name in self.label_name2tag:
                    return -1
                self.label_name2tag[label_i.label_name] = label_i.label_id
        if len(self.label_infos) == 0 or len(self.label_infos) != len(self.label_name2tag):
            return -1
        return 0

    def ParseInfoConfig(self, config_file):
        if os.path.exists(config_file) == False:
            return -1
        with open(config_file, 'rb') as handle:
            self.texts = pickle.load(handle)
            self.labels = pickle.load(handle)
            self.vectors = pickle.load(handle)
        print("database", len(self.texts), len(self.labels), len(self.vectors))
        if len(self.texts) == 0 or len(self.labels) == 0 or len(self.vectors) == 0:
            return -1
        if len(self.texts) != len(self.labels) or len(self.labels) != len(self.vectors):
            return -1
        return 0





