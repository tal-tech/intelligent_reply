# encoding:utf-8
import time
import logging
import traceback
import datetime as dt
from logging import handlers
from configparser import ConfigParser

def get_current_time_str():
    time_local = time.localtime()
    res = time.strftime('%Y_%m_%d_%H-%M-%S',time_local)
    return res

class Logger():
    def __init__(self, conf):
        levels = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
            "error": logging.ERROR
        }
        #print("--logger init...")
        try:
            cp = ConfigParser()

            cp.read(conf)

            time_str = get_current_time_str()
            self.filename = cp.get('log', 'FILE_NAME_PREFIX') + '_' + time_str + '.log'
            self.print_level = levels[cp.get('log', 'LEVEL').strip()]
            self.when = cp.get("log", "WHEN")
            self.backCount = cp.getint("log", "BACK_COUNT")
            #self.fmt = cp.get("log", "FMT")
            #self.fmt = '[%(levelname)s %(asctime)s  %(process)d %(filename)s:%(funcName)s:%(lineno)s] %(message)s'
            self.fmt = '%(asctime)s [%(process)d] %(levelname)s %(funcName)s:%(lineno)s - msg: %(message)s'
            self.logger = logging.getLogger(self.filename)

            formatter = MyFormatter(fmt=self.fmt)
            # self.fmt = '[%(levelname)s]%(asctime)s - %(filename)s:%(lineno)d - %(message)s'
            #print('\n'.join(['--%s:%s' % item for item in self.__dict__.items()]))
            self.logger.setLevel(self.print_level)  # 设置日志级别

            ## 写文件
            # if cp.getint("log","WRITE_FILE") == 1:
            #     th = handlers.TimedRotatingFileHandler(filename=self.filename,
            #                                            when=self.when,
            #                                            backupCount=self.backCount,
            #                                            encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
            #     # 实例化TimedRotatingFileHandler
            #     # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
            #     # S 秒
            #     # M 分
            #     # H 小时
            #     # D 天
            #     # W 每星期（interval==0时代表星期一）
            #     # midnight 每天凌晨
            #     th.setFormatter(formatter)  # 设置文件里写入的格式
            #     self.logger.addHandler(th)

            sh = logging.StreamHandler()  # 往屏幕上输出
            sh.setFormatter(formatter)  # 设置屏幕上显示的格式
            self.logger.addHandler(sh)  # 把对象加到logger里

            self.logger.info("{} init successful!".format(self.__class__.__name__))

        except:
            print("{} init fail, detail{}".format(self.__class__.__name__, traceback.format_exc()))

class MyFormatter(logging.Formatter):
    converter=dt.datetime.fromtimestamp
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s.%03d" % (t, record.msecs)
        return s

if __name__ == "__main__":
    logger = Logger('data/log.conf')
    logger.logger.info('sadd')
