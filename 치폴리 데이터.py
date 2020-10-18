#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


chipo=pd.read_excel('C:/datapy/chipotle.xlsx',sheet_name=0)


# In[5]:


# 실행 결과 값을 통해 index는 0부터 4621까지 총 4,622개의 행과 Data columns는 'order_id', 'quantity', 'item_name', 
#'choice_description', item_price 5개의 컬럼으로 이루어져있는것을 알 수 있습니다. 
#5개의 컬럼 중 2개는 int 타입 3개는 object 타입으로 구성되어있는것을 알 수 있습니다.
print(chipo.shape)
print('-----------------------')
print(chipo.info())


# In[7]:


chipo.head(10)


# In[8]:


#행, 열 데이터 확인하기

#column과 index를 확인.
print(chipo.columns)
print('-'*100)
print(chipo.index)

#실행 결과를 통해 columns 명과 data type, index의 범위와 step을 알 수 있습니다.

#여기서 quantity와 item_price는 연속형 피처입니다. 


# In[9]:


#describe() 함수를 이용하여 기초 통계량 출력하기

chipo['order_id'] = chipo['order_id'].astype(str)
print(chipo.describe())


# In[11]:


chipo.describe()


# In[12]:


# unique() 함수를 이용하여 개수 파악하기
print(len(chipo['order_id'].unique())) #order_id의 개수를 출력
print(len(chipo['item_name'].unique())) #item_name의 개수를 출력


# In[13]:


#가장 많이 주문한 아이템 Top 10 출력하기

#item_name의 value_counts를 높은숫자부터 상위 10개만 출력하는 코드를 작성합니다.

item_count = chipo['item_name'].value_counts()[0:10]
for idx, (val,cnt) in enumerate(item_count.iteritems(),1):
    print("Top", idx, ":", val, cnt)


# In[14]:


#아이템별 주문 개수와 총량

#groupby() 함수를 이용하여 아이템별 주문 개수와 총량을 구해봅니다.

#groupby() 함수는 데이터 프레임에서 특정 피처를 기준으로 그룹을 생성하며 이를 통해 그룹별 연산을 적용 할 수 있습니다.

#아이템별 주문개수 출력하는 코드
order_count = chipo.groupby('item_name')['order_id'].count()
order_count[:10]


# In[16]:


#아이템별 주문 총량 계산하는 함수
item_quantity = chipo.groupby('item_name')['quantity'].sum()
item_quantity[0:10]


# In[17]:


#시각화로 분석 결과 살펴보기

#tolist()와 Numpy를 활용하여 x_pos를 선언하고 0부터 50까지의 숫자를 그래프의 x축으로 사용합니다. 
#y축에는 주문 총량에 해당하는 item_quantity.values.tolist()를 넣습니다.

import numpy as np
import matplotlib.pyplot as plt

item_name_list = item_quantity.index.tolist()
x_pos = np.arange(len(item_name_list))
order_cnt = item_quantity.values.tolist()

plt.bar(x_pos, order_cnt, align= 'center')
plt.ylabel('ordered_item_count')
plt.title('Distribution of all ordered item')

plt.show()


# In[30]:


#데이터 전처리 : 전처리 함수 사용하기

#item_price 피처의 요약 통계를 구하기 위해 데이터 전처리를 합니다.

#전처리 전 지금 현재 상태를 확인해보도록 하겠습니다.

print(chipo.info())
print('-'*100)
chipo['item_price'].head()


# In[33]:


#가격의 앞에 붙은 $ 기호를 제거하기 위해 apply 함수와 lambda 함수를 이용하여 제거합니다.
chipo['item_price'] = chipo['item_price'].apply(lambda x : float(x[1:]))

chipo.describe()


# In[ ]:


#apply() 함수는 시리즈 단위 연산을 처리하는 기능을 수행하며, sum()이나 mean()과 같이 연산이 정의된 함수를 파라미터로 받습니다. 따라서 첫 번째 문자열을 제거한 뒤 나머지 문자열을 수치형으로 바꿔주는 함수를 파라미터로 입력할 수도 있습니다.


# In[34]:


#주문당 평균 계산금액 출력하기

#order_id 로 그룹을 생성한 뒤, item_price 피처에 sum() 함수를 적용하고 mean() 함수를 추가하여 평균 금액을 계산할 수 있습니다.

chipo.groupby('order_id')['item_price'].sum().mean()


# In[36]:


#한 주문에 10달러 이상 지불한 주문 번호(id) 출력하기

#order_id 피처를 기준으로 그룹을 만들어 quantity, item_price 피처의 합계를 계산하고 결과값이 10 이상인 값을 필터링 합니다

chipo_orderid_group = chipo.groupby('order_id').sum()
results = chipo_orderid_group[chipo_orderid_group.item_price >= 10]
print(results[:10])
print(results.index.values)


# In[37]:


#각 아이템의 가격 구하기

#① chipo.quantity == 1으로 동일 아이템을 1개만 구매한 주문을 선별합니다.

#② item_name을 기준으로 groupby 연산을 수행한뒤, min() 함수로 각 그룹별 최저가를 계산합니다.

#③ item_price를 기준으로 정렬하는 sort_values() 함수를 적용합니다.

chipo_one_item = chipo[chipo.quantity == 1]
price_per_item = chipo_one_item.groupby('item_name').min()
price_per_item.sort_values(by = 'item_price', ascending = False)[:10]


# In[38]:


#각 아이템의 대략적인 가격을 2개의 그래프로 시각화하여 나타낼 수 있습니다.

#이를 통해 2~4달러, 혹은 6~8달러 정도에 아이템의 가격대가 형성되어 있음을 알 수 있습니다.

#아이템 가격 분포 그래프
item_name_list = price_per_item.index.tolist()
x_pos = np.arange(len(item_name_list))
item_price = price_per_item['item_price'].tolist()

plt.bar(x_pos, item_price, align = 'center')
plt.ylabel('item price($)')
plt.title('Distribution of item price')
plt.show()

#아이템 가격 히스토그램
plt.hist(item_price)
plt.ylabel('counts')
plt.title('Histogram of item price')
plt.show()


# In[39]:


#가장 비싼 주문에서 아이템이 총 몇 개 팔렸는지 구하기

#order_id에 그룹별 합계 연산을 적용하고 item_price를 기준으로 sort_values를 반환하면 가장 비싼 주문순으로 연산 결과를 얻을 수 있습니다.
chipo.groupby('order_id').sum().sort_values(by = 'item_price', ascending = False)[:5]


# In[40]:


#'Veggie Salad Bowl'이 몇 번 주문되었는지 구하기

#특정 아이템인 'Veggie Salad Bowl' 이 몇 번이나 주문 되었는지 알아보겠습니다.
#필터링을 이용하여 drop_duplicates() 함수를 사용합니다. 이는 한 주문내에서 item_name이 중복 집계된 경우를 제거해주기 위함입니다.

chipo_salad = chipo[chipo['item_name'] == 'Veggie Salad Bowl']

chipo_salad = chipo_salad.drop_duplicates(['item_name', 'order_id'])

print(len(chipo_salad))
chipo_salad.head(5)


# In[41]:


#위에 사용했던 방법과 비슷한 방법으로 'Chicken Bowl'을 2개 이상 주문한 주문 횟수를 구합니다.

#'Chicken Bowl'을 주문한 데이터만을 필터링하고 주문 번호를 그룹으로 선정하여 sum() 함수를 적용한 결과에서 quantity를 선택하면 
#이 결과가 각 주문마다 'Chicken Bowl'을 주문한 횟수를 의미합니다.

chipo_chicken = chipo[chipo['item_name'] == 'Chicken Bowl']
chipo_chicken_ordersum = chipo_chicken.groupby('order_id').sum()['quantity']
chipo_chicken_result = chipo_chicken_ordersum[chipo_chicken_ordersum >= 2]

print(len(chipo_chicken_result))
chipo_chicken_result.head(5)


# In[ ]:




