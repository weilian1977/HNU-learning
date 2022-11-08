# -*- coding:utf-8 -*-

import logging
import time

class CustomFormatter(logging.Formatter):

    blue = "\x1b[32;20m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt=None):
        self.FORMATS = {
            logging.DEBUG:
            CustomFormatter.grey + fmt + CustomFormatter.reset,
            logging.INFO:
            CustomFormatter.blue + fmt + CustomFormatter.reset,
            logging.WARNING:
            CustomFormatter.yellow + fmt + CustomFormatter.reset,
            logging.ERROR:
            CustomFormatter.red + fmt + CustomFormatter.reset,
            logging.CRITICAL:
            CustomFormatter.bold_red + fmt + CustomFormatter.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


#记录器
my_logger = logging.getLogger("my_logger")
my_logger.setLevel(logging.DEBUG)

#处理器
_sh1 = logging.StreamHandler()
_fh1 = logging.FileHandler(
    filename="%s.log"% time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) , mode='w'
)  # 没有设置输出级别，将用my_logger的输出级别(并且输出级别在设置的时候级别不能比Logger的低!!!)，设置了就使用自己的输出级别

# 格式器
_fmt = "%(asctime)s-%(levelname)-9s- %(filename)s:%(lineno)s - %(message)s"
_sh1.setFormatter(CustomFormatter(_fmt))
_fh1.setFormatter(logging.Formatter(_fmt))

#记录器设置处理器
my_logger.addHandler(_sh1)
my_logger.addHandler(_fh1)

if __name__ == '__main__':
    #打印日志代码
    my_logger.debug("This is  DEBUG of my_logger !!")
    my_logger.info("This is  INFO of my_logger !!")
    my_logger.warning("This is  WARNING of my_logger !!")
    my_logger.error("This is  ERROR of my_logger !!")
    my_logger.critical("This is  CRITICAL of my_logger !!")
