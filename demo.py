import json
from algorithm_process import AlgorithmOperation
# from src.Engine import Engine

# engine = Engine("data/main.conf")
# print(engine)

def main(algorithm_input):
    # output_str = engine.Process(algorithm_input)
    # print(output_str)
    res, algorithm_output = AlgorithmOperation.process(algorithm_input)
    if res == 0:
        print(json.dumps(algorithm_output))
    else:
        print('algorithm phonetic error! %s',res)
                        

if __name__=="__main__":
    algorithm_input = {
        "query": "如何上课？",
        "result_count": 1
    }
    # print(json.dumps(algorithm_input,indent=4))
    main(algorithm_input)
    # pass
