# _*_ coding:utf-8 _*_
# 开发人员: LingJiabin
# 开发工具: PyCharm
# 开发日期: 2020/3/23 下午 2:15
# 文件名  : demo3.py

import os


f_pos = open(r"D:\py38\pycharm\S_pos.txt", "w", encoding='utf-8')
f_neg = open(r"D:\py38\pycharm\S_neg.txt", "w", encoding='utf-8')
count = 0
text = ''
with open(r'D:\py38\pycharm\s1.txt', 'r', encoding='utf-8') as fr:
    temp = fr.readline()
    while temp:
        while count != 10:
            text = text + ' ' + temp.strip()
            temp = fr.readline()
            count += 1
        temp = fr.readline()
        line = "__label__正" + "\t" + text + "\n"
        f_pos.write(line)
        count = 0
        text = ''

count = 0
text = ''

with open(r'D:\py38\pycharm\s0.txt', 'r', encoding='utf-8') as fr:
    temp = fr.readline()
    while temp:
        while count != 10:
            text = text + ' ' + temp.strip()
            temp = fr.readline()
            count += 1
        temp = fr.readline()
        line = "__label__负" + "\t" + text + "\n"
        f_neg.write(line)
        count = 0
        text = ''


