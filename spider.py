import requests
import urllib
import csv
import os
from urllib import parse
import shutil  # 用来删除文件夹

# 构建宠物小精灵英中字典，key是英文，value是中文
def petslist():
    file = './pokemonlist.csv'
    pets = {}
    with open(file)as f:
        rows = csv.reader(f)
        for row in rows:
            pets[row[3]] = row[1]
    return pets

#获取数据集文件夹名称
def dataname():
    path = './dataset'
    classes = os.listdir(path)
    return classes

#获取搜狗页面指定小精灵名称图片，获取前四页
def getImage(pet,path):
    basepath = './collections'
    nowpath = os.path.join(basepath,path)
    if os.path.exists(nowpath):
        shutil.rmtree(nowpath)
    os.mkdir(nowpath)
    #构造URL
    baseurl = "https://pic.sogou.com/pics?"
    headers = {'User-Agent': 'Mozilla/5.0'}
    params = {
        'mode' : '1',
        'reqType':'ajax',
        'tn':'0',
        'reqFrom':'detail',
        'query':pet,
        'start':1
    }
    m = 0
    for start in range(1,193,48):
        params['start'] = start
        res = requests.get(baseurl,headers=headers,params=params).json()
        # 得到图片信息列表
        imgs_items = res['items']
        # 存储每个想要保存的图片链接，为了后续
        for i in imgs_items:
            # thumbUrl存储的图片是大小为480*360的图片网页
            img_url = i['thumbUrl']
            print('*********' + str(m) + '.png********' + 'Downloading...')
            print('下载的url: ', img_url)
            # 下载图片并且保存
            urllib.request.urlretrieve(img_url, nowpath+"/"+path+str(m) + '.jpg')
            m = m + 1
    print('Download complete !')
    

if __name__ == '__main__':
    #print(dataname())
    dataset = dataname()
    # 指定获取的列表，注释掉后使用前面全部文件夹列表
    dataset = ['Mewtwo','Pikachu','Charmander','Bulbasaur','Squirtle',
    'Psyduck','Spearow','Fearow','Dratini','Aerodactyl','Rapidash',
    'Shellder','Ninetales','Pidgey','Machamp','Mankey','Muk','Sandslash',
    'Raichu','Lapras']
    labels = petslist()
    #批量获取
    for item in dataset:
        #排除的列表
        if item in ["Zapdos",'Kadabra','Omanyte','Shellder','Bellsprout']:
            pass
        else:
            if labels.__contains__(item):
                getImage(labels[item],item)
                print(item, labels[item])
            else:
                print(item)
    #单个获取 参数为搜索的关键词，批量保存的文件名
    # getImage("口袋妖怪 拉普拉斯","Shellder")
    