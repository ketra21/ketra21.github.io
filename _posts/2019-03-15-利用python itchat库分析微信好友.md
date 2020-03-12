---
layout:     post
title:      利用itchat库分析微信好友
subtitle:   
date:       2019-03-15
author:     阳光和鱼
header-img: img/post-bg-desk.jpg
catalog: true
tags:
    - python
	- 数据分析

---

- # 利用itchat库分析微信好友

  

  在这个项目中，通过开源的微信个人号接口 [itchat](http://itchat.readthedocs.io/zh/latest/) 来实现 Python 调用微信好友数据，并做一些有趣的统计和分析。
  
  ## 目录
  
  1. 向文件传输助手发送信息
  2. 统计微信好友的男女比例
  3. 分析微信好友的地域分布
  4. 生成不同性别好友的微信签名词云图
  5. 对不同性别好友的签名进行情感分析
  
  ## 0. 登陆并发送打招呼信息
  
  ### 登陆
  
  首先导入itchat包，并调用login()函数登陆网页微信，并扫描二维码以登陆网页微信。
  
  ```python
  # 导入项目中所需要的包
  #coding:utf-8
  
  import pandas as pd
  import re
  import os 
  import numpy as np
  import pinyin
  import matplotlib.pyplot as plt
  import seaborn as sb
  import itchat
  
  plt.rcParams['font.sans-serif'] = ['SimHei']
  %matplotlib inline
  
  print("所有库导入成功！")
  # 调用login()函数以登录网页微信
  itchat.login()
  Getting uuid of QR code.
  Downloading QR code.
  Please scan the QR code to log in.
  #不选择自己的账号信息
  dataset = itchat.get_friends(update=True)[1:]
  打个招呼
  ```
  
  ### 打个招呼
  
  调用itchat的send()函数向文件传输助手filehelper发送一个打招呼信息吧！你需要完成以下内容： – 将想要发送的信息内容赋值给message
  
  ```python
  # 将信息内容赋值给message
  message = "我在使用Python发送信息！"
  # 发送消息
  itchat.send(message, 'filehelper')
  <ItchatReturnValue: {'BaseResponse': {'Ret': 0, 'ErrMsg': '请求成功', 'RawMsg': '请求成功'}, 'MsgID': '7371139409964672208', 'LocalID': '15364822585397'}>
  ```
  
  打开手机微信端的文件传输助手，看看是否收到了这条信息。
  
  ```python
  ### 退出登陆
  itchat.logout()
  LOG OUT!
  
  <ItchatReturnValue: {'BaseResponse': {'ErrMsg': '请求成功', 'Ret': 0, 'RawMsg': 'logout successfully.'}}>
  数据整理
  ```
  
  ### 数据整理
  
  ```python
  ### dataset选择部分列
  def preprocess_data(dataset):
  
      data = [{'NickName':item['NickName'],
   'Sex':item['Sex'],
   'Province':item['Province'],
   'City':item['City'],
   'Signature':item['Signature'],
   'KeyWord':item['KeyWord'],
  } 
              for item in dataset]
  
      return data
  
  pre_data = preprocess_data(dataset)
  df_all = pd.DataFrame()
  for i in range(len(pre_data)):
      df = pd.DataFrame([pre_data[i]],index=[pre_data[i]['NickName']])
      df_all =df_all.append(df)
  df_all.columns
  Index(['City', 'KeyWord', 'NickName', 'Province', 'Sex', 'Signature'], dtype='object')
  df = df_all[['NickName', 'Sex', 'Province', 'City', 'Signature', 'KeyWord']]
  ```
  
  ## 1. 好友男女比例
  
  根据我们希望探索的问题，需要从数据集中取出以下几个部分： – NickName：微信昵称 – Sex：性别，1表示男性，2表示女性 – Province：省份 – City：城市 – Signature：微信签名
  
  ### 练习：统计男女比例
  
  - 统计好友性别，分为男性、女性与未知三种，赋值到已经定义好的sex字典中。
  
  ```python
  ### 输出好友性别数量
  sex = df['Sex'].value_counts()
  print("我共有好友{0}人，其中男性：{1}人,女性{2}人，未注明性别{3}人。".format(np.sum(sex),sex[1],sex[2],sex[0]))
  我共有好友1217人，其中男性：669人,女性468人，未注明性别80人。
  #绘制好友性别圆饼图
  plt.figure(figsize=(5,5), dpi=80)
  plt.axes(aspect=1) 
  plt.pie([sex[1], sex[2]],
          labels=['男性','女性'],
          labeldistance = 1.1,
          autopct = '%3.1f%%',
          shadow = False,
          startangle = 90,
          pctdistance = 0.6 
  )
  
  #plt.legend(loc='upper left',)，此处显示标签
  plt.title("我的微信好友性别比例")
  plt.show()
  ```
  
  ![img](https://github.com/ketra21/picbed/raw/master/picgo/20200312204709wechat_sex_pert.png)
  
  ## 2. 好友地域分布
  
  ### 统计好友省份
  
  ```python
  #去除省份为空值的数据
  province = df[df['Province'] != '']['Province'].value_counts()
  province = province.sort_values(ascending= False)
  province.index
  province_list = [
      '江苏', '上海', '北京', '河北', '广东', '浙江', '山东', '安徽', '河南', '湖北',
      '福建', '天津', '湖南', '云南', '江西', '广西', '山西', '辽宁', '黑龙江', '陕西',
      '四川',  '吉林','新疆', '西藏', '重庆','海南','甘肃','贵州']
  #粗略假定国内省份为汉字，港澳台、国外地区为字母。
  province_guonei = []
  province_guowai= []
  for x in province.index:
      if x[0].encode( 'UTF-8' ).isalpha():
          province_guowai.append(x)
      else:
          province_guonei.append(x)
  print('好友地域分布：国内:',province[province_guonei].sum(),'人  国外:',province[province_guowai].sum(),'人')
  好友地域分布：国内: 911 人  国外: 65 人
  province[province_guonei][10:].sum()
  111
  qita = pd.Series([111,65]).rename({0:'其他省份',1:'国外'})
  qita
  其他省份    111
  国外       65
  dtype: int64
  province_new = province[province_guonei].head(10).append(qita)
  # plot chart
  plt.figure(figsize=(6,6))
  province_new.plot(kind='pie', autopct='%1.1f%%',startangle=90, shadow=False, legend = False, fontsize=14);
  ```
  
  ![img](https://github.com/ketra21/picbed/raw/master/picgo/20200312204804_wechat_province.png)
  
  City列同样可以进行分析。
  
  ## 3. 生成好友个性签名词云图
  
  在这里我们希望生成词云，只需要调用第三方库即可，Python有大量的库可以使用，能极大提高开发效率，是编程入门的绝佳选择。为了比较不同性别词云差异，将筛选男、女两种签名数据。
  
  ```python
  #得到男女个性签名
  df1 =['','','']
  df1[1] = df[df['Sex'] ==1]['Signature'].values.tolist()
  df1[2] = df[df['Sex'] ==2]['Signature'].values.tolist()
  #df1 = pd.DataFrame({'Signature':data1[1]})
  ### 生成词云自定义函数
  def signature_wordcloud(data,jpg_n):
  ​```
  输入为待分析的数据集和性别识别编号。
  参数：
      data——数据集；
      jpg_n——1/2，分别代表男/女
  输出为该数据集下的使用对应性别背景图片词云。
  ​```    
      from wordcloud import WordCloud
      import jieba
      tList = []
      for i in range(data.shape[0]):
          signature = data['Signature'][i].replace(" ", "").replace("span", "").replace("class", "").replace("emoji", "")
          rep = re.compile("1f\d.+")
          signature = rep.sub("", signature)
          if len(signature) > 0:
              tList.append(signature)
      text1 = "".join(tList)
  
      wordlist_jieba = jieba.cut(text1, cut_all=True)
      wl_space_split = " ".join(wordlist_jieba)
      import PIL.Image as Image
  
      jpg_open = ["man.jpg","woman.jpg"]
      alice_coloring = np.array(Image.open(jpg_open[jpg_n-1]))
  
      my_wordcloud = WordCloud(background_color="white", max_words=2000, mask=alice_coloring,
                               max_font_size=50, random_state=42, font_path='./SimHei.ttf').generate(wl_space_split)
  
      #plt.figure(figsize=[12,12])
      plt.imshow(my_wordcloud)   
      plt.axis("off")
  def Signature_sex(data):
  ​```
  输入为包含数据集data
  输出为两种性别的个性签名词云。
  ​```
  
      import matplotlib.pyplot as plt
      import numpy.random as rnd
  
      plt.figure(figsize=[12,12])
      plt.rcParams['font.size'] =12
      #man
      df1 = data[data['Sex'] ==1]['Signature'].values.tolist()
      df1 = pd.DataFrame({'Signature':df1})
      plt.subplot(1,2,1)
      signature_wordcloud(df1,1)
      plt.axis("off")
      plt.text(200,600,'男性')
      #woman
      df2 = data[data['Sex'] ==2]['Signature'].values.tolist()
      df2 = pd.DataFrame({'Signature':df2})
      plt.subplot(1,2,2)
      signature_wordcloud(df2,2)
      plt.axis("off")
      plt.text(200,600,'女性')
  
      plt.show()
  Signature_sex(df)
  ```
  
  ![img](https://github.com/ketra21/picbed/raw/master/picgo/20200312204846_wechat_sex_cloud.png)
  
  从两性个性签名词云可以看出，男性关键字最多的有：一切、自己、一天、人生、不要等等；女性关键字最多的有：自己、努力、就是、没有、快乐、人生等等。自己、人生是男女共有的。但我的朋友以年轻人为主不好好拼搏，谈什么人生~~
  
  ## 4. 对好友签名进行情感分析
  
  在这部分内容中，我们调用了[SnowNLP](https://github.com/isnowfy/snownlp)的情感分析，它是一个python写的类库，可以方便的处理中文文本内容，不用我们实现其中具体的代码。一般来说，情感分析的目的是为了找出作者观点的态度，是正向还是负向，或者更具体的，我们希望知道他的情绪。我们简单地假设大于0.66表示积极，低于0.33表示消极，其他表示中立。
  
  ## 练习：统计好友签名情感分析结果比例
  
  - 统计sentiments中**大于0.66**的个数
  - 统计sentiments中**大于等于0.33且小于等于0.66**的个数
  - 统计sentiments中**小于0.33**的个数
  
  ## 男性:
  
  ```python
  signature_1 = []
  for i in df[df['Sex']==1]['Signature']:
      if i != '':
          signature_1.append(i)
  ### 以下内容无需修改，直接运行即可
  from snownlp import SnowNLP
  
  sentiments_1 = []
  for i in signature_1:
      sentiments_1.append(SnowNLP(i).sentiments)
  ### TODO：统计sentiments中大于0.66的个数
  positive_1 = sum(i >0.66 for i in sentiments_1)
  
  ### TODO：统计sentiments中大于等于0.33且小于等于0.66的个数
  neutral_1 = sum(i >=0.33 for i in sentiments_1)-  sum(i >0.66 for i in sentiments_1)
  
  ### TODO：统计sentiments中小于0.33的个数
  negative_1 = sum(i <0.33 for i in sentiments_1)
  print('男性好友签名情感积极:',positive_1,'中性:',neutral_1,'消极:',negative_1)
  男性好友签名情感积极: 285 中性: 146 消极: 74
  emotion_1 =[positive_1, neutral_1, negative_1]
  emotion_1 = emotion_1 / np.sum(emotion_1)
  emotion_1
  array([0.56435644, 0.28910891, 0.14653465])
  ```
  
  ## 女性:
  
  ```python
  signature = []
  for i in df[df['Sex']==2]['Signature']:
      if i != '':
          signature.append(i)
  ### 以下内容无需修改，直接运行即可
  sentiments = []
  for i in signature:
      sentiments.append(SnowNLP(i).sentiments)
  ### TODO：统计sentiments中大于0.66的个数
  positive = sum(i >0.66 for i in sentiments)
  
  ### TODO：统计sentiments中大于等于0.33且小于等于0.66的个数
  neutral = sum(i >=0.33 for i in sentiments)-  sum(i >0.66 for i in sentiments)
  
  ### TODO：统计sentiments中小于0.33的个数
  negative = sum(i <0.33 for i in sentiments)
  print('女性好友签名情感积极:',positive,'中性：',neutral,'消极：',negative)
  女性好友签名情感积极: 196 中性： 107 消极： 57
  emotion =[positive, neutral, negative]
  emotion = emotion / np.sum(emotion)
  emotion
  array([0.54444444, 0.29722222, 0.15833333])
  index = np.arange(3)
  subjects =['积极','中性','消极']
  X = np.arange(3)+1 #X是1,2,3柱的个数
  plt.bar(X, emotion_1, alpha=0.9, width = 0.25, facecolor = 'blue', edgecolor = 'white', label='男', lw=1)
  plt.bar(X+0.25, emotion, alpha=0.9, width = 0.25, facecolor = 'red', edgecolor = 'white', label='女', lw=1)
  plt.legend(loc="upper right"); # label的位置在右上
  plt.xticks(index+1.125, subjects);
  plt.rcParams['font.size'] = 12
  ```
  
  ![img](https://github.com/ketra21/picbed/raw/master/picgo/20200312204958_wechat_feel.jpg)
  
  从图上可以看出男性、女性的个性签名在情绪上以积极为主，约占55%，中性约占30%，消极约占15%。在两性对比上，男女差别不大，男性积极情绪略高。
