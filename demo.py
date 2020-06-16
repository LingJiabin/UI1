# _*_ coding:utf-8 _*_
# 开发人员: LingJiabin
# 开发工具: PyCharm
# 开发日期: 2020/3/11 13:49
# 文件名  : demo.py

from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox
from PySide2.QtCore import QFile
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCharts import *
from PySide2.QtGui import *

import os
import re
import sys
import jieba.analyse
import linecache
#import requests
import collections
import you_get
from datetime import datetime
import moviepy.editor as me

import test2 as t2


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.metrics import classification_report


class News_A:

    def __init__(self):

        self.ui = QUiLoader().load('Main.ui')

        self.indir_src = r'src' + '\\'
        self.indir_data = r'data' + '\\'
        self.src_dir_num = len(os.listdir('src'))

        self.files = os.listdir(self.indir_src)  # 返回指定文件夹的列表名
        self.files.sort(key=lambda x:int(x))
        for src_Names in self.files:
            self.ui.textEdit.append(src_Names)

        if os.path.exists('config.txt'):
            with open('config.txt', 'r', encoding='utf-8') as f:
                self.ui.lineEdit_4.setText(f.readline().strip())
                self.ui.lineEdit_5.setText(f.readline().strip())

        self.A_Pid = QtCharts.QPieSeries()
        self.A_Pid1 = QtCharts.QPieSeries()
        self.All_Pid = QtCharts.QPieSeries()
        self.All_Pid1 = QtCharts.QPieSeries()

        self.Bar = QtCharts.QBarSeries()
        self.QX = QtCharts.QBarCategoryAxis()
        self.QY = QtCharts.QValueAxis()

        self.Chart_P = QtCharts.QChart()
        self.Chart_B = QtCharts.QChart()

        self.ui.pushButton.clicked.connect(self.A_News)
        self.ui.pushButton_2.clicked.connect(self.All_News)
        self.ui.pushButton_3.clicked.connect(self.News_search)
        self.ui.pushButton_4.clicked.connect(self.Get_V)
        self.ui.pushButton_5.clicked.connect(self.Train_test)
        self.ui.pushButton_6.clicked.connect(self.Save_Apid)
        self.ui.pushButton_7.clicked.connect(self.Show_APid)
        self.ui.pushButton_15.clicked.connect(self.Show_AllPid)
        self.ui.pushButton_16.clicked.connect(self.Show_Bar)

    # 文本已经分析过寻找分析文件
    def Find_Txt(self, file_path):
        A_list = []
        S_list = []
        with open(file_path, 'r', encoding='utf-8') as fr:
            for x in range(14):
                A_list.append(fr.readline().replace('\n',''))
            t_Name = fr.readline().replace('\n','')

            S_list.append(fr.readline().replace('\n', ''))
            S_list.append(fr.readline().replace('\n', ''))

            S_lable = fr.readline().replace('\n','')
            texts = fr.readline().replace('\n','')

            return  A_list, t_Name, S_list, S_lable, texts

    # 文本未分析过寻找新闻文件
    def Analysis_Txt(self, file_path, file_name):

        # 对应标签数字化
        label_map = {'__label__财经': 1, '__label__彩票': 2, '__label__房产': 3, '__label__股票': 4, '__label__家居': 5,
                     '__label__教育': 6, '__label__科技': 7, '__label__社会': 8, '__label__时尚': 9, '__label__时政': 10,
                     '__label__体育': 11, '__label__星座': 12, '__label__游戏': 13, '__label__娱乐': 14}


        classifier = joblib.load("clf_model.m")  # 加载训练模型
        tf = joblib.load('tf_model.m')

        S_classifier = joblib.load("S_clf_model.m")
        S_tf = joblib.load('S_tf_model.m')

        # 输出测试数据
        t_name = []
        t_num = []

        # 构建tag_name
        for i in label_map.keys():
            t_name.append(i.strip('__label__'))
            t_num.append(int(label_map[i]))

        Num_map = dict(zip(t_num, t_name))
        # print(Num_map.get(10))


        with open(file_path, 'r', encoding='utf-8') as fr:
            file_text = fr.read()

        #筛选关键词
        key_word = jieba.analyse.extract_tags(file_text.replace("\t", "").replace("\n", ""), topK=20, withWeight=False,
                                          allowPOS=('n','nr','ns', 'nt', 'nz', 'q', 'tg'))

        #筛选情感词
        S_word = jieba.analyse.extract_tags(file_text.replace("\t", "").replace("\n", ""), topK=20, withWeight=False,
                                              allowPOS=('n', 'nz', 'a', 'an', 'e', 'i', 'v', 'vi'))

        outline = ' '.join(key_word)  # 提取新闻文本关键词
        S_line = ' '.join(S_word)

        tests = []
        tests.append(outline)

        S_tests = []
        S_tests.append(S_line)

        S_proba = S_classifier.predict_proba(S_tf.transform(S_tests))

        print(S_proba)
        print(S_classifier.predict(S_tf.transform(S_tests)))

        s0 = float(S_proba[0][0])
        s1 = float(S_proba[0][1])

        s0 = round(s0 + 0.005, 2)
        s1 = round(s1 + 0.005, 2)

        S_res = round(abs(s0 - s1), 2)
        S_lable = ''
        if S_res >= 0.37:
            if s0 < s1:
                S_lable = "积极的"
                print("正")
            else:
                S_lable = "消极的"
                print("负")
        else:
            S_lable = "中立的"
            print("中")

        S_list = []
        S_list.append('消极 ' + str(s0))
        S_list.append('积极 ' + str(s1))

        print(tests)
        sum = 0.0
        proba = classifier.predict_proba(tf.transform(tests))

        for x in range(14):
            num = str(proba[0][x])
            sum += float(num)

        A_list = []
        for i in range(14):
            num = str(proba[0][i])
            name = str(Num_map.get(i+1))
            #print(type(num))
            persent = int((float(num)/sum)*100 + 0.5)
            A_list.append(name + ' ' + str(persent) + '%')

        print(proba)
        pre_num = int(classifier.predict(tf.transform(tests)))
        print('预测结果为: %s' % Num_map.get(pre_num))
        #self.ui.textEdit_3.append('文章分类结果:'+ Num_map.get(pre_num) +'\n')
        #self.ui.textEdit_3.append('文章关键词:'+ tests[0] )

        t_Name =  Num_map.get(pre_num)
        texts = tests[0]

        #创建一个data文件,并写入分析数据
        path = self.indir_data + file_name
        if not os.path.exists(path):
            os.mkdir(path)
        detail_txt = path + '\Detail.txt'

        with open(detail_txt, 'w', encoding='utf-8') as f:
            for x in A_list:
                f.write(x + '\n')

            f.write(t_Name + '\n')
            f.write(S_list[0] + '\n')
            f.write(S_list[1] + '\n')
            f.write(S_lable + '\n')
            f.write(texts)

        return A_list, t_Name, S_list, S_lable, texts

    #单独分析
    def A_News(self):
        self.ui.textEdit_2.clear()
        self.ui.textEdit_2.setPlainText('请稍等....')

        file_Name = self.ui.lineEdit.text()
        if file_Name == '':
            Box = QMessageBox()
            Box.setWindowTitle('提示')
            Box.setText('请输入文件名')
            Box.exec_()
            return 0

        flag = 0

        d_files = os.listdir(self.indir_data)  # 返回指定文件夹的列表名
        d_files.sort(key=lambda x:int(x))

        for data_Names in d_files:
            if data_Names == file_Name:
                flag = 1
                break;
            else:
                flag = 0

        if flag == 1:
            fp = 'data' + '\\' + data_Names + r'\Detail.txt'
            A_list, t_Name, S_list, S_lable, texts = self.Find_Txt(file_path=fp)
        else:
            fp = 'src' + '\\' + file_Name + r'\News.txt'
            A_list, t_Name, S_list, S_lable, texts = self.Analysis_Txt(file_path=fp, file_name=file_Name)

        temp = []
        self.ui.textEdit_2.clear()

        self.A_Pid.clear()
        self.A_Pid1.clear()
        self.All_Pid.clear()
        self.All_Pid1.clear()

        for x in A_list:
            self.ui.textEdit_2.append(x)
            temp = x.split()
            temp_mun = temp[1].split('%')
            if int(temp_mun[0]) > 5:
                self.A_Pid.append(temp[0], int(temp_mun[0]))
            #print(type(temp_mun))

        print(S_list)
        for x in S_list:
            temp = x.split()
            self.A_Pid1.append(temp[0], float(temp[1]))

        self.A_Pid.setLabelsVisible()
        self.A_Pid1.setLabelsVisible()
        self.ui.textEdit_2.append('\n文章分类结果:'+ t_Name)
        self.ui.textEdit_2.append('\n文章情感极性:' + S_lable)
        self.ui.textEdit_2.append('\n文章关键词:'+ texts)

        print("A_News ok")

    #整体分析
    def All_News(self):

        label_key = {'财经': 1, '彩票': 2, '房产': 3, '股票': 4, '家居': 5,
                     '教育': 6, '科技': 7, '社会': 8, '时尚': 9, '时政': 10,
                     '体育': 11, '星座': 12, '游戏': 13, '娱乐': 14}

        S_key = {'消极的': 0, '积极的': 1, '中立的': 2}


        count = 0
        sta = [0 for x in range(14)] #新闻标签为列表的
        sta1 = [0 for x in range(3)] #情感

        s_files = os.listdir(self.indir_src)  # 返回指定文件夹的列表名
        s_files.sort(key=lambda x:int(x))
        self.ui.textEdit_3.append('正在计算...')

        for s_Name in s_files:
            path = 'data'+ '\\' + s_Name
            if os.path.exists(path):
                fp = 'data' + '\\' + s_Name + r'\Detail.txt'
                A_list, t_Name, S_list, S_lable, texts = self.Find_Txt(file_path=fp)
                sta[label_key[t_Name]-1] += 1
                sta1[S_key[S_lable]] += 1
            else:
                fp = 'src' + '\\' + s_Name + r'\News.txt'
                A_list, t_Name, S_list, S_lable, texts = self.Analysis_Txt(file_path=fp, file_name=s_Name)
                sta[label_key[t_Name]-1] += 1
                sta1[S_key[S_lable]] += 1

            count += 1

        self.ui.textEdit_3.clear()
        count = float(count)
        key = list(label_key.keys())
        S_Lable_k = list(S_key.keys())

        self.A_Pid.clear()
        self.A_Pid1.clear()
        self.All_Pid.clear()
        self.All_Pid1.clear()

        #统计分类结果
        num = 0
        for x in range(14):

            num = sta[x]/count + 0.005
            num = int(num*100)

            self.ui.textEdit_3.append(key[x]+':'+ str(num) + "%")
            if num > 3.0:
                self.All_Pid.append(key[x], num)

        #统计情感极性
        for x in range(3):
            num = sta1[x]/count + 0.005
            num = int(num * 100)
            self.ui.textEdit_3.append(S_Lable_k[x] + ':' + str(num) + "%")

        #开始统计热点词
        data_list = os.listdir('data')
        file = open('cache.txt', 'w', encoding='utf-8')

        for x in data_list:
            path = r'data' + '\\' + x + r'\Detail.txt'
            file.write(linecache.getline(path, 19))

        file.close()

        with open('cache.txt', 'r', encoding='utf-8') as f:
            word = f.read().split()

        print(word)
        #print('%s' % collections.Counter(word))
        word_res = collections.Counter(word)

        flag = 0
        self.ui.textEdit_3.append('\n热点词汇' + '\t' + '出现次数')
        for y in word_res.most_common(10):  #选出前十的个数
            self.ui.textEdit_3.append(y[0] + '\t' + str(y[1]))
            self.All_Pid1.append(y[0], y[1])
            '''
            flag += 1
            if flag == 10:
                break
            '''


        self.ui.textEdit_3.append('\n视频总数: ' + str(count))



        print("All_News ok")

    #检索
    def News_search(self):

        label_key = {'财经': 1, '彩票': 2, '房产': 3, '股票': 4, '家居': 5,
                     '教育': 6, '科技': 7, '社会': 8, '时尚': 9, '时政': 10,
                     '体育': 11, '星座': 12, '游戏': 13, '娱乐': 14}

        key = self.ui.textEdit_4.toPlainText()
        if key == '':
            Box = QMessageBox()
            Box.setWindowTitle('提示')
            Box.setText('请输入关键词')
            Box.exec_()
            return 0

        self.ui.textEdit_5.clear()

        key_words = key.split()     #转化为list
        key_num = len(key_words)

        files = os.listdir(self.indir_data)
        files.sort(key=lambda x:int(x))
        dir_num = len(files)

        sub = {i:0 for i in files}
        for x in key_words:
            for y in files:

                file_path = self.indir_data  + y + r'\Detail.txt'
                D_keys = linecache.getline(file_path, 19).split()

                for z in D_keys:
                    if z == x:
                        sub[y] += 1
                        break;

        Max_dir = '0'
        Max_num = 0
        self.ui.textEdit_5.append('相关文件如下' + '\t' + '类型\n')

        Analay = {i:0 for i in label_key.keys()}
        #print(Analay)
        self.Bar.clear()

        for k in sub:
            if sub[k] > 0:
                path = self.indir_data + k + r'\Detail.txt'
                k_lable = linecache.getline(path, 15).strip()
                #print(k)
                self.ui.textEdit_5.append(k + '\t\t' + k_lable)
                #设置条形图
                set = QtCharts.QBarSet(str(k))
                num = round(sub[k]/key_num, 1)
                set.append(num)

                Analay[k_lable] += num

                self.Bar.append(set)
                if sub[k] > Max_num:
                    Max_num = sub[k]
                    Max_dir = k
                    Max_lable = k_lable
        #print(sub)

        #分析用户可能性
        Max = 0
        for k in Analay.keys():
            if Analay[k] > Max:
                Max = Analay[k]
                possible_lable = k


        self.ui.textEdit_5.append('\n最大相关文件为' + '\t' + '类型\n')
        self.ui.textEdit_5.append(Max_dir + '\t\t' + Max_lable)
        self.ui.textEdit_5.append('\n详细信息如下')
        path = self.indir_data + Max_dir + r'\Detail.txt'
        self.ui.textEdit_5.append('\n情感极性:\n' + linecache.getline(path, 18).strip())
        self.ui.textEdit_5.append('\n积极消极占比: \n' + linecache.getline(path, 16) + linecache.getline(path, 17).strip())
        self.ui.textEdit_5.append('\n关键词:\n' + linecache.getline(path, 19).strip())

        self.ui.textEdit_5.append('\n来源url:')
        path = self.indir_src + Max_dir + r'\url.txt'
        with open(path, 'r', encoding='utf-8') as fr:
            temp = fr.read().replace('\n', '')
            temp1 = temp.split('//', 1)
        self.ui.textEdit_5.append('https://' + temp1[1])


        self.ui.textEdit_5.append('\n你要找的类型可能为:')
        self.ui.textEdit_5.append(possible_lable)
        self.Bar.setLabelsVisible()

        self.QX.append(" ")
        self.QY.setRange(0, 1)
        print("search_News ok")

    #视频获取
    def Get_V(self):

        def Download(url, file_path):
            sys.argv = ['you-get', '-o', file_path, '-O', 'News', url]
            you_get.main()

        def Translate_V(file_path):
            if os.path.exists(file_path):
                file_path_V = file_path + r'\News.mp4'
                file_path_A = file_path + r'\News.wav'

                # print(file_path_V)
                # print(file_path_A)

                video = me.VideoFileClip(file_path_V)
                audio = video.audio
                audio.write_audiofile(file_path_A)

            else:
                print('文件不存在')

        # 输入的url
        # url = input('输入地址:')
        url = self.ui.lineEdit_6.text().strip()
        if url == '':
            Box = QMessageBox()
            Box.windowTitle('提示')
            Box.setText('请输入url')
            Box.exec_()
            return 0

        # 构建文件目录

        Apid = self.ui.lineEdit_4.text().strip()
        Sk = self.ui.lineEdit_5.text().strip()

        if (Apid == '' or Sk == ''):
            A_Box = QMessageBox()
            A_Box.setText(u'请输入讯飞api')
            A_Box.exec_()
            return 0

        self.src_dir_num += 1 #文件个数


        file_path = self.indir_src  + str(self.src_dir_num) #重要

        print(file_path)

        if not os.path.exists(file_path):
            os.mkdir(file_path)
        else:
            os.system('cd ' + file_path + r' && del /q *')
            print("文件存在")

        Box = QMessageBox()
        Box.setWindowTitle("提示")
        Box.setText('时间较长，请稍等...')
        Box.exec_()

        Download(url, file_path)
        Translate_V(file_path)

        url_txt = file_path + r'\url.txt'
        with open(url_txt, 'w', encoding='utf-8') as f:
            f.write(url)

        #apid = "5e5267cc"
        #sk = "1b0ac63fa1dca15f894e6a427022a70f


        self.ui.textEdit_6.clear()
        self.ui.textEdit_6.append('下载完成')
        try:
            t2.Translate_A(apid=Apid, sk=Sk, pre_fp=file_path)
        except:
            print("讯飞失效")

        with open(file_path + r'\News.txt', 'r', encoding='utf-8') as fr:
            file_text = fr.read()

        file_text = file_text.replace("\t", "").replace("\n", "")
        self.ui.textEdit_6.append('\n视频内容:')
        self.ui.textEdit_6.append(file_text)

        self.ui.textEdit_6.append('\n关键词:')
        keyW = jieba.analyse.extract_tags(file_text, topK=20, withWeight=False, allowPOS=('n','nr','ns', 'nt', 'nz', 'q', 'tg'))
        self.ui.textEdit_6.append(' '.join(keyW))

        self.ui.textEdit_6.append('\n视频来源:')
        url = url.split('/', 4)
        self.ui.textEdit_6.append(url[2])
        self.ui.textEdit_6.append('\n获取时间:')
        self.ui.textEdit_6.append(datetime.now().strftime('%Y-%m-%d'))

        print("Get_News ok")

    #训练数据

    #新闻分类训练
    def Train_test(self):
        file_path = self.ui.lineEdit_3.text().strip()
        T_Box = QMessageBox()

        if not os.path.exists(file_path):
            T_Box.setText('文件不存在')
            T_Box.exec_()
        else:
            T_Box.setText('训练中请稍等')
            T_Box.exec_()

        # 导入清洗好的数据集和测试集
        df_news = pd.read_table(file_path, names=['label', 'content'], sep='\t', encoding='utf-8')
        df_news = df_news.dropna()

        # 对应标签数字化
        label_map = {'__label__财经': 1, '__label__彩票': 2, '__label__房产': 3, '__label__股票': 4, '__label__家居': 5,
                     '__label__教育': 6, '__label__科技': 7, '__label__社会': 8, '__label__时尚': 9, '__label__时政': 10,
                     '__label__体育': 11, '__label__星座': 12, '__label__游戏': 13, '__label__娱乐': 14}

        # 保存原始标签
        news_label = list(df_news['label'].unique())
        # 替换标签
        df_news['label'] = df_news['label'].map(label_map)
        news_num = list(df_news['label'].unique())
        # print(df_news.head())

        # 划分训练集和测试集
        # x 为测试数据 y 为测试结果
        #x_train, x_test, y_train, y_test = train_test_split(df_news['content'].values, df_news['label'].values, random_state=1, stratify=df_news['label'].values)

        def create_words(data):  # 对数据进行进一步清洗变成 list to list格式
            words = []
            for index in range(len(data)):
                try:
                    words.append(data[index])
                except Exception:
                    print(index)
            return words

        # 将训练集和测试集的类型进一步处理
        #train_words = create_words(x_train)
        train_words = create_words(df_news['content'].values)
        train_y = create_words(df_news['label'].values)
        #test_words = create_words(x_test)

        classifier = MultinomialNB()
        tf = TfidfVectorizer()
        vc = tf.fit(train_words)
        #tf.fit(test_words)
        #classifier.fit(tf.transform(train_words), y_train)
        classifier.fit(tf.transform(train_words), train_y)
        #classifier.fit(tf.transform(test_words), y_test)
        joblib.dump(classifier, "clf_model.m")  # 保存训练模型
        joblib.dump(tf, 'tf_model.m')

        # 查看特征矩阵
        '''
        vc_fit = tf.fit_transform(train_words)
        print(tf.get_feature_names())
        print(vc_fit.toarray())
        print(vc_fit.toarray().sum(axis=0))
        '''
        # vector = tf.transform(train_words)  # 对文本进行标记并建立索引

        # print(tf.vocabulary_)  # 建立结果
        # print(vector.shape)  # 输出(文章个数, 索引个数(特征个数))
        # print(tf.get_feature_names())  # 索引名称
        # print(vector)
        # print(vector.toarray())#特征矩阵
        # print(vector.toarray().sum(axis=0))#出现次数总的和

        # 对训练模型进行分类测试
        # 对测试模型进行分类测试
        '''
        score_train = float(classifier.score(tf.transform(train_words), y_train)) * 100
        score_test = float(classifier.score(tf.transform(test_words), y_test)) * 100
        print('训练集正确率:%.4f %%' % score_train)
        print('测试集正确率:%.4f %%' % score_test)
        y_pd = classifier.predict(tf.transform(test_words))
        '''

        with open('News.txt' , 'r', encoding='utf-8') as fr:
            file_text = fr.read()

        key_word = jieba.analyse.extract_tags(file_text.replace("\t", "").replace("\n", ""), topK=20, withWeight=False,
                                          allowPOS=('n','nr','ns', 'nt', 'nz', 'q', 'tg'))
        outline = ' '.join(key_word)  # 提取新闻文本关键词
        # print(outline,type(outline))
        tests = []
        tests.append(outline)
        p = classifier.predict_proba(tf.transform(tests))
        print(p)

        self.Train_S()

        T_Box = QMessageBox()
        T_Box.setText('训练完成')
        T_Box.exec_()
        print('训练完成')
   
    #情感极性训练
    def Train_S(self):


        file_path = 'S_train.txt'

        label_map = {'__label__负': 0, '__label__正': 1}
        df_S = pd.read_table(file_path, names=['label', 'content'], sep='\t', encoding='utf-8')
        df_S = df_S.dropna()

        df_S['label'] = df_S['label'].map(label_map)

        train_words = create_words(df_S['content'].values)
        train_y = create_words(df_S['label'].values)

        classifier = MultinomialNB()
        tf = TfidfVectorizer()
        tf.fit(train_words)

        for i in range(25):
            classifier.fit(tf.transform(train_words), train_y)

        joblib.dump(classifier, "S_clf_model.m")  # 保存训练模型
        joblib.dump(tf, 'S_tf_model.m')

    #存储用户APi模块
    def Save_Apid(self):

        if self.ui.checkBox.isChecked():
            Apid = self.ui.lineEdit_4.text().strip()
            Sk = self.ui.lineEdit_5.text().strip()
            if (Apid == '' or Sk == ''):
                Box = QMessageBox()
                Box.setText(u'不能为空')
                Box.setWindowTitle('提示')
                Box.exec_()
                return 0
            with open('config.txt', 'w', encoding='utf-8') as f:
                f.write(Apid+'\n')
                f.write(Sk)

                Box = QMessageBox()
                Box.setText(u'记住成功')
                Box.setWindowTitle('提示')
                Box.exec_()
        elif self.ui.checkBox_2.isChecked():
            os.remove('config.txt')
            Box = QMessageBox()
            Box.setText(u'忘记成功')
            Box.setWindowTitle('提示')
            Box.exec_()

        print('保存成功')


    def Show_APid(self):

        self.A_News()
        self.A_Pid.setPieSize(0.8)
        self.A_Pid.setHoleSize(0.6)
        self.A_Pid1.setPieSize(0.4)
        self.Chart_P.addSeries(self.A_Pid)
        self.Chart_P.addSeries(self.A_Pid1)
        self.Chart_P.setTitle('类别百分比')

        self.ChartView_P = QtCharts.QChartView(self.Chart_P)
        self.ChartView_P.setRenderHint(QPainter.Antialiasing)
        self.ChartView_P.resize(800, 800)
        self.ChartView_P.setWindowTitle('A_News_Pid')
        self.ChartView_P.show()

    def Show_AllPid(self):

        self.All_News()

        self.All_Pid.setPieSize(0.8)
        self.All_Pid.setHoleSize(0.6)
        self.All_Pid.setLabelsVisible()
        self.All_Pid1.setPieSize(0.4)
        self.All_Pid1.setLabelsVisible()

        self.Chart_P.addSeries(self.All_Pid)
        self.Chart_P.addSeries(self.All_Pid1)
        self.Chart_P.setTitle('类别百分比')

        self.ChartView_P = QtCharts.QChartView(self.Chart_P)
        self.ChartView_P.setRenderHint(QPainter.Antialiasing)
        self.ChartView_P.resize(1200, 800)
        self.ChartView_P.setWindowTitle('All_News_Pid')
        self.ChartView_P.show()

    def Show_Bar(self):

        self.News_search()
        self.Chart_B.addSeries(self.Bar)
        self.Chart_B.setTitle('相关图')
        self.Chart_B.addAxis(self.QX, Qt.AlignBottom)
        self.Bar.attachAxis(self.QX)
        self.Chart_B.addAxis(self.QY, Qt.AlignLeft)
        self.Bar.attachAxis(self.QY)

        self.ChartView_B = QtCharts.QChartView(self.Chart_B)
        self.ChartView_B.setRenderHint(QPainter.Antialiasing)
        self.ChartView_B.resize(1200, 600)
        self.ChartView_B.setWindowTitle('The_Bar')
        self.ChartView_B.show()



app = QApplication([])
A = News_A()
A.ui.setWindowOpacity(0.9)
A.ui.show()
app.exec_()