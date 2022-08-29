import pandas as pd
from datetime import datetime
import numpy as np
import time
from bs4 import BeautifulSoup
import re

'''
chorme webdriver 실행하기
'''

from selenium import webdriver
driver = webdriver.Chrome(r"C:\Users\seolbluewings\Desktop\sample\chromedriver.exe")


'''
숫자를 입력받아서 page url을 완성시키는 함수 작성
'''

def get_search_page_url(page) :
    return "http://www.shinsegae-lnb.com/product/wine?currentPage={}&orderBy=2&listSize=12&selectedWineType=0&selectedWineNation=0&selectedSugar=0&searchText=#orderBy".format(page)



'''
사이트에 등록된 전체 와인상품 개수 구하기 
'''

page_url = get_search_page_url(1)
page_url = str(page_url)
driver.get(page_url)
time.sleep(1)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

tot_prd_cnt = int(re.findall('[0-9]+',str(soup.select('div.cont > p.total > span')))[0])


'''
for문으로 돌면서 각 와인상품에 대한 정보 긁어서 하나의 DataFrame으로 만들기
'''

for i in range(0, 100) :
    page_url = get_search_page_url(i+1)
    page_url = str(page_url)
    driver.get(page_url)
    time.sleep(10)
    
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    num_product_list = len(soup.select('div.cont > ul.list > li > div > a'))    
    
    for j in range(0,num_product_list) :
        product_info = str(soup.select('div.cont > ul.list > li > div > a')[j])
        url_info = re.findall(r'\/product+.+(?=" tabindex)',product_info)[0] # /product로 시작해서 tabindex 앞에 있는 것까지만 불러오기
        print(url_info)
        
        product_url = 'http://www.shinsegae-lnb.com'+str(url_info)
        
        driver.get(product_url)
        time.sleep(3)
        
        html2 = driver.page_source
        soup2 = BeautifulSoup(html2, 'html.parser')
            
        prd_nm   = soup2.select('div.box1 > dl > dt')[0].text.strip().replace(" ","_")
        prd_enm  = soup2.select('div.box1 > dl > dd')[0].text.strip().replace(" ","_")
        prd_type = soup2.select('div.box2 > ul > li.type1 > span')[1].text.strip()
        prd_breed = soup2.select('div.box2 > ul > li.type3 > span')[1].text.strip()
        
        if soup2.select('div.box3 > dl > dd > span.on') != [] : 
        
            if (
                re.findall('[ㄱ-ㅎ|가-힣]+', re.findall(r'\D+', soup2.select('div.box3 > dl > dd > span.on')[0].text.strip())[0])[0] == '당도'
            ) :
                prd_sgr = int(re.findall(r'\d', soup2.select('div.box3 > dl > dd > span.on')[0].text.strip())[1])
            elif (
                re.findall('[ㄱ-ㅎ|가-힣]+', re.findall(r'\D+', soup2.select('div.box3 > dl > dd > span.on')[0].text.strip())[0])[0]  == '산도' 
            ) :
                prd_acd = int(re.findall(r'\d', soup2.select('div.box3 > dl > dd > span.on')[0].text.strip())[1])
            elif (
                re.findall('[ㄱ-ㅎ|가-힣]+', re.findall(r'\D+', soup2.select('div.box3 > dl > dd > span.on')[0].text.strip())[0])[0]  == '바디'
            ) :
                prd_bdy = int(re.findall(r'\d', soup2.select('div.box3 > dl > dd > span.on')[0].text.strip())[1])
            else :
                prd_sgr = None
            
        
            if (
                re.findall('[ㄱ-ㅎ|가-힣]+', re.findall(r'\D+', soup2.select('div.box3 > dl > dd > span.on')[0].text.strip())[0])[0] == '당도' and
                re.findall('[ㄱ-ㅎ|가-힣]+', re.findall(r'\D+', soup2.select('div.box3 > dl > dd > span.on')[1].text.strip())[0])[0] == '산도' 
            ) :
                prd_acd = int(re.findall(r'\d', soup2.select('div.box3 > dl > dd > span.on')[1].text.strip())[1])
            elif (
                re.findall('[ㄱ-ㅎ|가-힣]+', re.findall(r'\D+', soup2.select('div.box3 > dl > dd > span.on')[0].text.strip())[0])[0] == '산도' and
                re.findall('[ㄱ-ㅎ|가-힣]+', re.findall(r'\D+', soup2.select('div.box3 > dl > dd > span.on')[1].text.strip())[0])[0] == '바디'
            ) :
                prd_bdy = int(re.findall(r'\d', soup2.select('div.box3 > dl > dd > span.on')[1].text.strip())[1])
            else :  
                prd_acd =None
        
            if (
                re.findall('[ㄱ-ㅎ|가-힣]+', re.findall(r'\D+', soup2.select('div.box3 > dl > dd > span.on')[0].text.strip())[0])[0] == '당도' and
                re.findall('[ㄱ-ㅎ|가-힣]+', re.findall(r'\D+', soup2.select('div.box3 > dl > dd > span.on')[1].text.strip())[0])[0] == '산도' and
                re.findall('[ㄱ-ㅎ|가-힣]+', re.findall(r'\D+', soup2.select('div.box3 > dl > dd > span.on')[2].text.strip())[0])[0] == '바디'
            ) :
                prd_bdy = int(re.findall(r'\d', soup2.select('div.box3 > dl > dd > span.on')[2].text.strip())[1])
            else :
                prd_bdy = None
        
        else : #추출되는 list가 null이라면, 즉 화면에서 당도,산도,바디 모두 체크가 되어있지 않다면
            prd_sgr = None; prd_acd = None; prd_bdy = None
        
        
        prd_txt  = soup2.select('div.box1 > dl > dd')[1].text.strip()
        prd_vol  = int(re.findall('[0-9]+',str(soup2.select('div.box2 > ul > li.type4 > span')[1]))[0])
        prd_match = soup2.select('div.box2 > ul > li.type5 > span')[1].text.strip()
        now = datetime.now()
        load_dh = now.strftime('%Y%m%d%H%M%S')
    
        df = pd.DataFrame({'PRD_NM' : [prd_nm],
                           'PRD_ENM' : [prd_enm],
                           'PRD_TYPE' : [prd_type],
                           'PRD_BREED' : [prd_breed],
                           'PRD_SGR' : [prd_sgr],
                           'PRD_ACD' : [prd_acd],
                           'PRD_BDY' : [prd_bdy],
                           'PRD_VOL' : [prd_vol],
                           'PRD_TXT' : [prd_txt],
                           'PRD_MATCH' : [prd_match],
                           'LOAD_DH' : [load_dh]})
    
        prd_df1 = prd_df1.append(df).reset_index(drop = True)

prd_df1.to_csv(r'C:\Users\seolbluewings\Desktop\sample\www_df.csv',sep=',',index = True)

