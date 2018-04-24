先对AVIRIS原始数据进行MNF变换，选取前15个波段，再利用CNN进行学习，样本精度到达0.99996！

代码在version10-20180406-1253-AVIRIS数据测试好.txt中，

模型存储在./oil_cnn model/fivecls_cnn.model-3.ckpt-250中

数据在./aviris_oil/mnf文件夹中
