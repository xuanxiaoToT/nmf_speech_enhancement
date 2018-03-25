"""NMF语音分离模型中，管理数据集与文件的模块

Authors :刘建东
"""
import os
import config as cnf

def find_files(speaker, fmt="wav"):
    files = os.listdir(cnf.path+speaker+"/")
    files = map(lambda x:cnf.path+speaker+"/"+x, files)
    files = filter(lambda x:fmt in x and speaker in x, files)
    return list(files)[:5]
