import re

import requests
import csv
from lxml import etree
import time
import multiprocessing
from multiprocessing import Manager
proxies= {'http':'175.42.128.21:9999','http':'125.108.124.19:9000','http':'125.110.93.153:9000','http':'106.42.44.96:9999','http':'218.204.153.156:8080'}
s=set()
counts = 0
def get_xiaoqu(index,p):#获取小区id信息
    head={
        'Host': 'sh.lianjia.com',
        'Referer': 'https://sh.lianjia.com/chengjiao/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
    }
    n=index*20+1
    l=list(range(n,n+20))
    for i in l:
        url='https://sh.lianjia.com/xiaoqu/pg'+str(i)
        try:
            r=requests.get(url=url,proxies=p,headers=head,timeout=3)
            html=etree.HTML(r.text)

            datas=html.xpath('//li[@class="clear xiaoquListItem"]/@data-id')
            title=html.xpath('//li[@class="clear xiaoquListItem"]/div[@class="info"]/div[@class="title"]/a/text()')
            # print('No:' + str(index), 'page:' + str(i))
            if(len(datas)==0):
                print(url)
                l.append(url)
            else:
                for data in datas:
                    s.add(data)
        except Exception as e:
            l.append(i)
            print(e)


def parse_xiaoqu(url, pa):
    head = {'Host': 'sh.lianjia.com',
            'Referer': 'https://sh.lianjia.com/chengjiao/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'

            }
    # print(url)
    r = requests.get(url, headers=head, proxies=proxies,timeout=5)
    html = etree.HTML(r.text)
    # print(r.text)
    num = html.xpath('//div[@class="total fl"]/span/text()')[0]
    num = int(num)
    datas = html.xpath('//li/div[@class="info"]')
    print('小区房源总数：', num, '第%d页房源数:' % pa, len(datas))
    if len(datas) == 0:
        return (num, [], 0)  # 服务器无返回数据，状态码返回0
    house_list = []
    for html1 in datas:
        title = html1.xpath('div[@class="title"]/a/text()')
        print(title[0])
        res=re.findall(r"\d+\.?\d*", title[0])
        titles=title[0].split()[0]
        if(len(res)<3):
            continue
        room=res[0]
        hall=res[1]
        area=res[2]

        info = html1.xpath('div[@class="address"]/div[@class="houseInfo"]/text()')
        floor = html1.xpath('div[@class="flood"]/div[@class="positionInfo"]/text()')
        info[0] = info[0].replace('\xa0', '')  # 该条信息中有个html语言的空格符号&nbsp；需要去掉，不然gbk编码会报错，gb18030显示问号
        date = html1.xpath('div[@class="address"]/div[@class="dealDate"]/text()')
        if(info[0].split()[-1]=='精装'):
            decoration=1
        elif(info[0].split()[-1]=='简装'):
            decoration=2
        elif(info[0].split()[-1]=='毛坯'):
            decoration=3
        else:
            decoration=4
        # 30天内成交的进入详情页面抓取
        if date[0] == '近30天内成交':
            p_url = html1.xpath('div[@class="title"]/a/@href')
            r = requests.get(p_url[0], headers=head, proxies=proxies,timeout=5)
            html = etree.HTML(r.text)
            price = html.xpath('//div[@class="overview"]/div[@class="info fr"]/div[@class="price"]/span/i/text()')
            unitprice = html.xpath('//div[@class="overview"]/div[@class="info fr"]/div[@class="price"]/b/text()')
            date = html.xpath('//div[@class="house-title LOGVIEWDATA LOGVIEW"]/div[@class="wrapper"]/span/text()')
            # 有的房源信息没有价格信息，显示暂无价格
            if len(price) == 0:
                price.append('暂无价格')
            if len(unitprice) == 0:
                unitprice.append('暂无单价')
            date[0] = date[0].replace('链家成交', '')
            a = [titles, room, hall, area,decoration, price[0], unitprice[0]]
            house_list.append(a)
            print(titles, room, hall, area,decoration, price[0], unitprice[0])
        else:
            price = html1.xpath('div[@class="address"]/div[@class="totalPrice"]/span/text()')
            unitprice = html1.xpath('div[@class="flood"]/div[@class="unitPrice"]/span/text()')
            if len(price) == 0:
                price = ['暂无价格']
            if len(unitprice) == 0:
                unitprice = ['暂无单价']
            a = [titles, room, hall, area,decoration, price[0], unitprice[0]]
            house_list.append(a)
            print(titles, room, hall, area,decoration, price[0], unitprice[0])
    print('                *********************         ', '第%d页完成！' % pa)
    return (num, house_list, 1)

def crow_xiaoqu(id):
    time.sleep(4)
    url='https://sh.lianjia.com/chengjiao/c%d/'%int(id)
    print(url)
    h_list=[]      #保存该小区抓取的所有房源信息
    fail_list=[]   #保存第一次抓取失败的页数，第一遍抓取完成后对这些页数再次抓取
    try:
        #爬取小区第一页信息
        result=parse_xiaoqu(url,1)
    except:
        #如果第一页信息第一次爬取失败，sleep2秒再次爬取
        time.sleep(2)
        result=parse_xiaoqu(url,1)
    #获取该小区房源总数num
    num = result[0]
    #如果无数据返回，sleep2秒再爬取一次
    if num == 0:
        time.sleep(2)
        result=parse_xiaoqu(url,1)
        num = result[0]
    new_list = result[1]
    pages=1
    for data in new_list:
        if data not in h_list:
            h_list.append(data)
    # 确定当前小区房源页数pages
    if num > 30:
        if num % 30 == 0:
            pages = num // 30
        else:
            pages = num // 30 + 1
    for pa in range(2,pages+1):
        new_url = 'https://sh.lianjia.com/chengjiao/pg'+str(pa)+'c'+str(id)
        try:
            result=parse_xiaoqu(new_url,pa)
            status=result[2]
            if status==1:
                new_list=result[1]
                #排重后存入h_list
                for data in new_list:
                    if data not in h_list:
                        h_list.append(data)
            else:
                fail_list.append(pa)
        except Exception as e:
            fail_list.append(pa)
            print(e)
    print('   开始抓取第一次失败页面')
    for pa in fail_list:
        new_url = 'https://sh.lianjia.com/chengjiao/pg' + str(pa) + 'c' + str(id)
        try:
            result = parse_xiaoqu(new_url,pa)
            status = result[2]
            if status == 1:
                new_list = result[1]
                for data in new_list:
                    if data not in h_list:
                        h_list.append(data)
            else:
                pass
        except Exception as e:
            print(e)
    print('    抓取完成，开始保存数据')
    #一个小区的数据全部抓完后存入csv
    with open('house_price.csv','a',newline='',encoding='utf-8-sig')as f:
        write=csv.writer(f)
        for data in h_list:
            write.writerow(data)
    #返回抓取到的该小区房源总数
    count=len(h_list)


if __name__ == '__main__':
    manager = Manager()
    process = []
    # for i in range(5):
    #     proxy=proxies
    #     p=multiprocessing.Process(target=get_xiaoqu,args=(i,proxy))
    #     process.append(p)
    #     p.start()
    # for p in process:
    #     p.join()
    #     # get_xiaoqu(i,proxy)
    #     # print('抓取完成')
    # with open('xiaoqu_id.csv','a',newline='',encoding='gb18030') as f:
    #     write = csv.writer(f)
    #     for data in s:
    #         write.writerow([data])
    #     f.close()
    id_list = []
    with open('xiaoqu_id.csv', 'r')as f:
        read = csv.reader(f)
        for id in read:
            id_list.append(id[0])
    m = 0
    # 可以通过修改range函数的起始参数来达到断点续抓
    for x in range(0, 3000):
        id=id_list[x]
        p=multiprocessing.Process(target=crow_xiaoqu, args=(id,))
        process.append(p)
        # crow_xiaoqu(id)
        # time.sleep(10)
        p.start()
    for p in process:
        time.sleep(10)
        p.join()



