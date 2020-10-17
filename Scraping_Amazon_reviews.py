#!/usr/bin/env python
# coding: utf-8

# In[182]:


'''
For scrapping task I have to scrap review of 10 phones with each having 10 reviews.

So,The testing will be done on total 100 reviews.
'''
import requests
from bs4 import BeautifulSoup  #I found beautiful soup more faster and easier to use than others.
import time


# In[183]:


start = time.process_time()


# In[184]:


#Providing all the links of phones review that I will scrap
url_links=[
     'https://www.amazon.in/OnePlus-Nord-Marble-128GB-Storage/product-reviews/B086977J3K/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
  'https://www.amazon.in/Apple-iPhone-Xs-512GB-Space/product-reviews/B07J3CJH8S/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
  'https://www.amazon.in/Samsung-Galaxy-Storage-Additional-Exchange/product-reviews/B089MQ7C7V/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
    ,'https://www.amazon.in/OPPO-Storage-Additional-Exchange-Offers/product-reviews/B086KF4FYC/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
    ,'https://www.amazon.in/Samsung-Galaxy-Ocean-128GB-Storage/product-reviews/B07HGGYWL6/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
    ,'https://www.amazon.in/Redmi-8A-Dual-Blue-Storage/product-reviews/B07X4R63DF/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
    ,'https://www.amazon.in/TECNO-Spark-Comet-Black-Storage/product-reviews/B08HX4RKR1/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
    ,'https://www.amazon.in/Apple-iPhone-11-64GB-White/product-reviews/B07XVMCLP7/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
    ,'https://www.amazon.in/Nokia-5-3-Android-Smartphone-64/product-reviews/B08GT28WQQ/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
    ,'https://www.amazon.in/Redmi-Note-Pebble-Grey-Storage/product-reviews/B086977TR6/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
]
url_names=['One Plus Nord 5G','iPhone Xs','Samsung Galaxy Note 20 Ultra 5G','oppo A11K',
           'Samsung Galaxy M31','Redmi 8A dual','Tecno Spark 6 Air','iPhone 11','Nokia 5.3 Android One Smartphone'
          ,'Redmi Note 9 (Pebble Grey, 4GB RAM 64GB Storage)']


# In[185]:


#This function will get the contents and remove duplicates,then returns cleaner ratings,review
def content_request(link):
    page=requests.get(link)
    soup=BeautifulSoup(page.content,'html.parser')
    rating=soup.find_all('i',class_='review-rating')
    review=soup.find_all('span',{"data-hook":"review-body"})
    reviews=cleaner(review)
    ratings=cleaner(rating)
    ratings.pop(0)
    ratings.pop(0)
    
    return ratings,reviews
                         
    


# In[186]:


#Used it to remove \n present in corpus
def cleaner(corpus):
    list_to_append=[]
    for i in range(0,len(corpus)):
        list_to_append.append(corpus[i].get_text())
        
    list_to_append[:]=[text.strip('\n') for text in list_to_append]
    
    return list_to_append


# In[187]:


#To make final Test corpus and Test labels
Test_corpus=[]
Test_labels=[]
count=0
for i in range(0,len(url_links)):
    link=url_links[i]
    rate,text=content_request(link)
    
    Test_corpus+=text
    Test_labels+=rate


# In[188]:


len(Test_corpus)
len(Test_labels)


# In[189]:


end=time.process_time()


# In[190]:


#Changing rating to only 0 or 1
Test_labels[:]=[t[0] for t in Test_labels]
for i in range(0, len(Test_labels)): 
    Test_labels[i] = int(Test_labels[i]) 
    if Test_labels[i]==1 or Test_labels[i]==2:
        Test_labels[i]=0
    else:
        Test_labels[i]=1


# In[191]:


difference=start-end


# In[194]:


print('Time taken to extract and clean information is-{}'.format(difference))


# In[ ]:





# In[ ]:




