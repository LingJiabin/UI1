# _*_ coding:utf-8 _*_
# 开发人员: LingJiabin
# 开发工具: PyCharm
# 开发日期: 2020/3/22 下午 8:52
# 文件名  : demo2.py

import jieba.analyse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.metrics import classification_report

file_path = r'D:\py38\pycharm\S_train.txt'

label_map = {'__label__负': 0, '__label__正': 1}
df_S = pd.read_table(file_path, names=['label', 'content'], sep='\t', encoding='utf-8')
df_S = df_S.dropna()

df_S['label'] = df_S['label'].map(label_map)
print(df_S.head())
print(list(label_map.keys()))
x_train, x_test, y_train, y_test = train_test_split(df_S['content'].values, df_S['label'].values,
                                                            random_state=1)

#print(df_S['content'].values)
#print(x_train)

def create_words(data):  # 对数据进行进一步清洗变成 list to list格式
    words = []
    for index in range(len(data)):
        try:
            words.append(data[index])
        except Exception:
            print(index)
    return words

train_words = create_words(df_S['content'].values)
train_y = create_words(df_S['label'].values)

test_words = create_words(x_test)
print(train_words)

classifier = MultinomialNB()
tf = TfidfVectorizer()
tf.fit(train_words)

for i in range(25):
    classifier.fit(tf.transform(train_words), train_y)
    #classifier.fit(tf.transform(test_words), y_test)

joblib.dump(classifier, "S_clf_model.m")  # 保存训练模型
joblib.dump(tf, 'S_tf_model.m')

# 提取新闻文本情感关键词
with open('News.txt', 'r', encoding='utf-8') as fr:
    file_text = fr.read()

key_word = jieba.analyse.extract_tags(file_text.replace("\t", "").replace("\n", ""), topK=20, withWeight=False,
                                      allowPOS=('n', 'nz', 'a', 'an', 'e', 'i', 'v', 'vi'))

#line = ' '.join(key_word)

line = '牵念 迁怒 谴 谴责 歉然 歉甚 瞧不起 瞧不上 瞧不上眼 切齿'
tests = []
tests.append(line)

proba = classifier.predict_proba(tf.transform(tests))
print(line)
print(proba)
print(classifier.predict(tf.transform(tests)))
s0 = float(proba[0][0])
s1 = float(proba[0][1])

s0 = round(s0+0.005, 2)
s1 = round(s1+0.005, 2)

print(s0)
print(s1)

S_res = round(abs(s0-s1),2)
print(S_res)

if S_res >= 0.37:
    if s0 < s1:
        print("正")
    else:
        print("负")
else:
    print("中")






