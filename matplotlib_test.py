import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

lena = mpimg.imread('C:\\test.jpg') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
#lena.shape #(512, 512, 3)
#print(lena[:,:,0])
print(lena.shape)
print(lena.dtype)

t=[[1,2,3],[4,5,6]]
plt.imshow(t) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()

'''
labels='frogs','hogs','dogs','logs'
sizes=15,20,45,10
colors='yellowgreen','gold','lightskyblue','lightcoral'
explode=0,0.1,0,0
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
plt.axis('equal')
plt.show()
'''