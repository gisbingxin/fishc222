from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

a=np.array([(255,255,30,40),(7,8,255,255)])
b=np.array([(20,30,40,50),(7,8,255,255)])
c=np.array([(3,4,5,6),(7,8,0,255)])
#print(a.shape)

temp1=np.empty((2,4,3))
#temp2=np.empty((3,2,4))
#print(temp1)

temp1[:,:,0]=a
temp1[:,:,1]=b
temp1[:,:,2]=c
temp2=temp1.astype(np.uint8)
print(temp2.dtype)

plt.imshow(temp2)  #利用数组转换，然后再imshow时，注意数据的类型，要转换为uint8
plt.show()
