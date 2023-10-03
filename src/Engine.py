# encoding:utf-8
import os
import logging
import sys
import json
from .Logger import Logger

from .Model import *


base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_path)

class Engine(object):
    def __init__(self, mainconf, loglevel="debug"):
        if loglevel == "info":
            level = logging.INFO
        elif loglevel == "debug":
            level = logging.DEBUG
        elif loglevel == " warning":
            level = logging.WARNING
        elif loglevel == "critical":
            level = logging.CRITICAL
        else:
            level = logging.WARNING

        cp = ConfigParser()
        cp.read(mainconf)
        self.logger = Logger(cp.get('log', 'log')).logger
        # logging.basicConfig(level=level, filename="text_match_faq.log", filemode="a", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.configfile = os.path.join(base_path, '../../', mainconf)
        self.Model = Model(cp.get('model', 'model'), self.logger)
        if self.Model.Init() != 0:
            self.logger.error("{} Init fail!!!!!!".format(self.__class__.__name__))

    def Process(self, input_str):
        self.logger.warning("get input! %s" % input_str)
        #################check input##################
        input = {}
        try:
            input = json.loads(input_str)
        except:
            self.logger.error("input json load failed!")
            output = self.ProcessOutPutFailedInfo(input, -1)
            self.logger.warning("get output! %s" % output)
            return output;

        status, info = self.checkInput(input)
        if status < 0:
            self.logger.error("check input json failed! %s" % info)
            output = self.ProcessOutPutFailedInfo(input, status)
            self.logger.warning("get output! %s" % output)
            return output;

        ############## process ##################
        slots = []
        parse_status = self.Parse(input["query"], slots)
        #################output for info##################
        if (parse_status < 0):
            output = self.ProcessOutPutFailedInfo(input, -50)
        else:
            output = self.ProcessOutPutInfo(input, slots)
        self.logger.warning("get output! %s" % output)
        return output;

    def Parse(self, input_str, slots):
        try:
            return self.Model.Match(input_str, slots)    
        except Exception as e:
            self.logger.error("get exception %s" % e)
            return -1

    def checkInput(self, input):
        if "query" not in input:
            self.logger.error("no query in input")
            return -2, "no query";
        if  isinstance(input["query"], str) == False:
            self.logger.error("query type error")
            return -2, "no query";

        if "result_count" not in input:
            input["result_count"] = 1
            return 0, "";
        elif  isinstance(input["result_count"], int) == False:
            self.logger.error("result_count type error")
            return -2, "result_count";
        return 0, "";

    def ProcessOutPutFailedInfo(self, input, status):
        output = dict()
        output["result_type"] = status
        output["parse_res"] = []
        return json.dumps(output, ensure_ascii=False)

    def ProcessOutPutInfo(self, input, slots):
        output = dict()
        output["result_type"] = 1

        last_output = []
        for slot in slots:
            class_label_i = {}
            class_label_i["question"] = slot.label_name
            class_label_i["confidence_score"] = slot.label_score
            class_label_i["question_id"] = int(slot.label_id)
            if len(last_output) < input["result_count"]:
                last_output.append(class_label_i)
        output["parse_res"] = last_output
        return json.dumps(output, ensure_ascii=False)







