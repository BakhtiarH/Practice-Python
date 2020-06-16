#!/usr/bin/env python
# coding: utf-8

# # Array

# In[ ]:


from array import*

arr = array('i',[1,2,3,4,5])
print(arr)


# In[ ]:


print(arr.buffer_info())


# In[ ]:


print(arr[2])


# In[ ]:


for i in arr:
    print(i)


# In[ ]:


for i in range(5):
    print(i,arr[i])


# In[ ]:


for i in range(1,3):
    print(i,arr[i])


# In[ ]:


arr.reverse()
print(arr)


# In[ ]:


arr.append(10)
print(arr)


# In[ ]:


arr.remove(2)
print(arr)


# In[ ]:


print(arr[3])


# In[ ]:


print(arr.index())


# In[ ]:


print(arr.index(3))


# In[4]:


from array import*
arr= array("i",[])
x = int(input("enter size of array"))
print("enter %d elements"%x)
for i in range(x):
    n=int(input())
    arr.append(n)
print(arr)


# # numpy
# 

# In[5]:


import numpy as np


# In[6]:


a= np.array([1,2,3])
a


# In[7]:


a[0]


# In[8]:


import time
import sys


# In[11]:


b=range(1000)
print(sys.getsizeof(5)*len(b))


# In[14]:


c=np.arange(1000)
print(c.size*c.itemsize)


# In[15]:


size = 100000


# In[16]:


L1=range(size)
L2=range(size)
A1=np.arange(size)
A2=np.arange(size) 


# In[19]:


start= time.time()
result = [(x+y) for x,y in zip(L1,L2)]
print(result)
print("PYTHON LIST TOOK:",(time.time()- start)*1000)


# In[20]:


start = time.time()
result= A1+A2
print("NUMPY ARRAY TOOK:",(time.time()- start)*1000)


# In[21]:


start = time.time()
result= A1+A2
print("NUMPY ARRAY TOOK:",(time.time()- start)*1000)


# In[22]:


a = np.array([[1,2],[3,4],[5,6]])
a


# In[24]:


a.ndim


# In[25]:


a.itemsize


# In[26]:


a.shape


# In[28]:


a= np.array ([[1,2],[3,4],[5,6]], dtype=np.float64)
a


# In[29]:


a.itemsize


# In[30]:


a= np.array ([[1,2],[3,4],[5,6]], dtype=np.complex)
a


# 
# 

# In[31]:


np.zeros((3,4))


# In[33]:


np.ones((3,4))


# In[36]:


np.ones((3,4))


# In[37]:


l=range(5)
l


# In[38]:


np.arange(5)


# In[40]:


print('concatenation example:')
print(np.char.add(['hello','hi'],['abc','xyz']))


# In[42]:


print(np.char.center('Hello',20, fillchar='-'))


# In[43]:


print(np.char.center('Hello',3))


# In[44]:


print(np.char.capitalize('Hello world'))


# In[45]:


print(np.char.title('how are you doing'))


# In[46]:


print(np.char.lower(['Hello','World']))
print(np.char.lower('Hello'))


# In[47]:


print(np.char.upper(['Hello','World']))
print(np.char.upper('Hello'))


# In[48]:


print(np.char.split('are you coming to the party'))


# In[49]:


print(np.char.lower(['Hello','World']))
print(np.char.lower('Hello'))


# In[50]:


print(np.char.strip(['nina,''admin','anaita'],'a'))


# In[52]:


print(np.char.join([':','-'],['dmy','ymd']))


# # Panadas

# In[53]:


import pandas as pd


# In[54]:


print(pd.__version__)


# # Series create,manipulate,querry,delete

# In[56]:


# creating a series from a list

arr=[0,1,2,3,4,]
s1= pd.Series(arr)
s1


# In[58]:


order=[1,2,3,4,5]
s2 = pd.Series(arr, index=order)
s2


# In[62]:


import numpy as np
n=np.random.randn(5)# Create a random Ndarray
index=['a','b','c','d','e']
s2=pd.Series(n, index=index)
s2


# In[63]:


# create series from dictionary
d={'a':1,'b':2,'c':3,'d':4,'e':5}
s3=pd.Series(d)
s3


# In[65]:


#modify the index of series
print(s1)
s1.index=['A','B','C','D','E']
s1


# In[66]:


# slicing
a=s1[:3]
a


# In[67]:


#append
s4= s1.append(s3)
s4


# In[68]:


s4.drop('e')


# # Series operations

# In[69]:


arr1=[0,1,2,3,4,5,7]
arr2=[6,7,8,9,5]


# In[71]:


s5=pd.Series(arr2)
s5


# In[73]:


s6 = pd.Series(arr1)
s6


# In[74]:


s5.append(s6)


# In[75]:


s5.add(s6)


# In[76]:


s5.sub(s6)


# In[86]:


s7 = s5.mul(s6)


# In[78]:


s5.div(s6)


# In[87]:


print('median',s7.median())
print('max',s7.max())
print('min',s7.min())


# In[88]:


s6


# # create Dataframe

# In[93]:


dates=pd.date_range('today',periods=6)# define time sequence as index
num_arr=np.random.randn(6,4)#Import numpy random array
columns=['A','B','C','D']# use the table as the column name

df1=pd.DataFrame(num_arr, index=dates, columns=columns)
df1


# In[95]:


#create dataframe with dictionary array

data={'animal':['cat','cat','snake','dog','dog','cat','snake','cat','dog','dog'],
     'age':[2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3],
     'visits':[1,3,2,3,2,3,1,1,2,1],
     'priority':['yes','yes','no','yes','no','no','no','yes','no','no']}
labels=['a','b','c','d','e','f','g','h','i','j']

df2 = pd.DataFrame(data, index=labels)
df2


# In[96]:


#see datatypes of array
df2.dtypes


# In[97]:


df2.head()


# In[98]:


df2.head(2)


# In[99]:


df2.tail()


# In[100]:


df2.index


# In[103]:


df2.animal


# In[105]:


df2.visits


# In[106]:


df2.columns


# In[107]:


df2.values


# In[108]:


df2.describe() #see statistical data of dataframe


# In[109]:


df2.T


# In[111]:


df2.columns


# In[112]:


df2.age


# In[113]:


df2.sort_values(by='age')


# In[114]:


#Slicing the dataframe
df2[1:3]


# In[116]:


df2.sort_values(by='age')[1:3]


# In[117]:


#query dataframe by tag
df2[['age','visits']]


# In[118]:


df2.iloc[1:3]#Query rows 2,3


# In[120]:


df3=df2.copy()
df3


# In[121]:


df3.isnull()


# In[126]:


df3.loc['f','age']=1.5
df3


# In[129]:


df3[['age',"visits"]].mean()


# In[131]:


df3['visits'].sum()


# In[132]:


df3.sum()


# In[137]:


string = pd.Series(['A','C','D','Aaa','BaCa',np.nan,'CBA','cow','owl'])
string.str.upper()


# # operations for dataframs missing values

# In[141]:


df4=df3.copy()
meanAge=df4['age'].mean()
df4['age'].fillna(4)


# In[142]:


df5=df3.copy()
df5.dropna(how='any')
df5


# # Dataframe file operations

# In[144]:


df3.to_csv('animal.csv')


# In[145]:


df_animal=pd.read_csv('animal.csv')
df_animal


# In[147]:


df3.to_excel('animal.xlsx',sheet_name='sheet1')
df_animal2=pd.read_excel('animal.xlsx','sheet1',index_col=None,na_values=['NA'])
df_animal2


# # Matplotlib

# In[149]:


from matplotlib import pylab
print(pylab.__version__)


# In[151]:


# use numpy to generate random data

import numpy as np
x=np.linspace(0,10,25)
y=x*x+2
print(x)
print(y)


# In[157]:


# it only takes one command to draw 

pylab.plot(x,y, "r")# r stand for red


# In[164]:


# drawing a subgraph

pylab.subplot(1,2,1)# the brackets contain row columns and indexes.
pylab.plot(x,y,'r--')# The third parameter here determines color and line style
pylab.subplot(2,2,2,)
pylab.plot(y,x,'g*-')
pylab.subplot(3,2,2)
pylab.plot(y,x,'g*-')


# # Operator Description
# 
# 

# In[167]:


from matplotlib import pyplot as plt


# In[170]:


fig = plt.figure()
from matplotlib.ticker import FuncFormatter
axis= fig.add_axes([0.5,0.1,0.8,0.8]) #control the left, right , width, height of the canvas (from 0 to1)
axis.plot(x,y, 'r')


# In[179]:


# subgraph

fig,axes= plt.subplots(nrows =1, ncols=2) #Submap is of 1 row, 2 columns
for ax in axes:
    ax.plot(x,y, 'r')


# In[182]:


# draw a picture or a graph in another graph

fig= plt.figure()
axes1 =fig.add_axes([0.1,0.1,0.8,1]) # big axes
axes2 =fig.add_axes([0.2,0.5,0.4,0.3])# small axes

axes1.plot(x,y, 'r')
axes2.plot(y,x, 'g')


# In[184]:


fig = plt.figure(figsize=(16,9),dpi=300) # New graphic object

fig.add_subplot

plt.plot(x,y,'r')


# In[191]:


fig,axes =plt.subplots()
axes.set_title("title")
axes.plot(x,y,'r')
axes.set_xlabel("x")
axes.set_ylabel('y')
axes.plot(x,y,'r')


# In[212]:


# in Matplotlib you can set other properties such as line color, transparency and more

fig,axes=plt.subplots(dpi=150)
axes.plot(x,x**3,color='red', alpha=0.5)
axes.plot(x,x*6, color="yellow",alpha=0.5)
axes.plot(x,x+3 ,color ='green',)


# In[216]:


fig, ax= plt.subplots(dpi=100)

ax.plot(x,x+1,'b',linewidth=0.25)
ax.plot(x,x+2,'b',linewidth=1)
ax.plot(x,x+3,'b',linewidth=0.25)
ax.plot(x,x+4,'b',linewidth=0.25)


# In[229]:


fig, ax= plt.subplots(dpi=100)

ax.plot(x,x+1,'b',lw=0.25,  linestyle='-')
ax.plot(x,x+2,'b',lw=1,     linestyle='-.')
ax.plot(x,x+3,'b',lw=0.25,  linestyle=':')
ax.plot(x,x+4,'b',lw=0.25,  linestyle='-')


line,= ax.plot(x,x+5,color='black',lw=1.5)
line.set_dashes([5,10,153,0])

line,= ax.plot(x,x+5,color='black',lw=1.5)
line.set_dashes([5,8,16,10])

line,= ax.plot(x,x+5,color='black',lw=1.5)
line.set_dashes([5,14,25,10])


# In[235]:


fig, Axes = plt.subplots(1,2, figsize=(10,5))

Axes[0].plot(x,x**2,x,x**3,lw=2)
Axes[0].grid(True)

Axes[1].plot(x,x**2,x,x**3,lw=2)
Axes[1].set_ylim([0,60])
Axes[1].set_xlim([2,5])
Axes[1].grid(True)


# # other 2d graphics

# In[250]:


n= np.array([0,1,2,3,4,5])
fig, axes= plt.subplots(1,4,figsize=(16,5))
axes[0].set_title('scatter plot')
axes[0].scatter(x,x+0.25*np.random.randn(len(x)))
axes[1].set_title('step')
axes[1].step(x,x+0.25*np.random.randn(len(x)))
axes[2].set_title('bar')
axes[2].bar(n,n**2,align='center',width=0.5,alpha=0.5)
axes[3].set_title("fill_between")
axes[3].fill_between(x,x**2,x**3,color='green',alpha=0.5)


# In[258]:


fig=plt.figure(figsize=(6,6))
ax= fig.add_axes([0.0,0.0,0.6,0.6], polar=True)
t = np.linspace(0,2*np.pi,100)
ax.plot(t,t*0.4, color='b',lw=3)


# In[263]:


# Draw a histogram

n =np.random.randn(100000)
fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].set_title('Default histogram')
axes[0].hist(n)

axes[1].set_title("Cumlative detailed histogram")
axes[1].hist(n,cumulative= True, bins=5)


# In[270]:


# Draw contour image

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

delta = 0.025
x= np.arange(-3.0,3.0,delta)
y=np.arange(-2.0,2.0,delta)
X,Y =np.meshgrid(x,y)
Z1=np.exp(-X**2 - Y**2)
Z2=np.exp(-(X-1)**2 - (Y-1)**2)
Z= (Z1-Z2)*2

print(x)
print(y)


# In[269]:


fig,ax = plt.subplots()
CS= ax.contour(X,Y,Z)
ax.clabel(CS, inline=1, fontsize=10)


# In[ ]:


import glob
all_data = pd.DataFrame()

