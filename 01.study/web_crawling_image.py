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
1. 이미지 따올 url 강제 입력
2. Chorm Driver에 url 입력
'''
url = 'https://www.naracellar.com/wine/wine_view.php?num=373&qstr='
driver.get(url)

'''
BeautifulSoup으로 html parsing 
'''

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

soup2 = soup.select('div.itemThumb > div.wine_swiper > ul.swiper-wrapper > li.swiper-slide > img')


'''
1. 비어있는 list 생성
2. 이미지가 여러개 있는 경우도 있어 list append
3. list에서 unique한 이름만 불러옴
'''

img_list = []
for i in range(0,len(soup2)) :
    img_list.append(soup2[i]['src'])  
unq_img_list = list(set(img_list))

'''
1. unique한 image 이름 개수만큼 for문 수행
2. 강제로 image 명칭을 url에 심어주고
3. 저장할 공간 지정 & 명칭 지정하여
4. 지정된 명칭으로 이미지 다운로드 
'''


for i in range(0,len(unq_img_list)) :
    prd_img_url = 'https://naracellar.com'+unq_img_list[i]
    imgName = r'C:\Users\seolbluewings\Desktop\www\img\prd_nm{}.jpg'.format(i)
    
    urllib.request.urlretrieve(prd_img_url,imgName)
    print("DownLoad Image Done !!")