import os

import numpy
import json

# from app import app, g_config
# from app.common.error_msg import Status, MyExcept
from src.Engine import Engine

engine = Engine("data/main.conf")
print(engine)

class AlgorithmOperation:
    @staticmethod
    def process(algorithm_input):
        print("alg_input:{}".format(algorithm_input))
        try:
            input_str = json.dumps(algorithm_input)
            output_str = engine.Process(input_str)
            print("alg_output:{}".format(output_str))
            alg_output_info = json.loads(output_str)
            parse_res = alg_output_info.get("parse_res")
            alg_result = list()
            for alg_element in parse_res:
                question_elem = dict()
                question = alg_element["question"]
                question_id = str(alg_element["question_id"])
                question_elem["question"] = question
                question_elem["question_id"] = question_id
                question_elem["confidence_score"] = float(alg_element["confidence_score"])
                alg_result.append(question_elem)
            return 0,{"result":alg_result}
        except Exception as e:
            return -1,{}
