# _*_ coding:utf-8 _*_
# 开发人员: LingJiabin
# 开发工具: PyCharm
# 开发日期: 2020/3/28 下午 12:27
# 文件名  : demo4.py

import collections
import  os
import linecache

data_list = os.listdir('data')
file = open('cache.txt', 'w', encoding='utf-8')


for x in data_list:
    path = r'data' + '\\' + x + r'\Detail.txt'
    file.write(linecache.getline(path, 19))

file.close()

with open('cache.txt', 'r', encoding='utf-8') as f:
    word = f.read().split()

kk = collections.Counter(word)
kkk = kk.most_common(10)
print(word)
#print('%s' % collections.Counter(word))
print(kkk[0][1])
print()






