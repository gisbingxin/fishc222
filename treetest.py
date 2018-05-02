import  numpy as np

a=True
if a:
    b=100
else:
    b=1
print(b)

# total_per_class=[100,200,300]
# #total_per_class=np.array(total_per_class)
#
# num_per_class=[35,23,34]
# class_num=3
# index_per_class=[]
# list0=[[12,34],[11],[1,2,3,4,5]]
# print(np.shape(list0),type(list0))
# for i in range(0,class_num):
#     #index_per_class[i]=np.random.random(1,total_per_class[i],size=num_per_class[i])
#     index_per_class.append(np.random.randint(1,total_per_class[i],size=num_per_class[i]))
# print((index_per_class[0]))
# '''compute confusion matrix
# labels.txt: contain label name.
# predict.txt: predict_label true_label
# '''
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import numpy as np
# #load labels.
# labels = []
# file = open('labels.txt', 'r')
# lines = file.readlines()
# for line in lines:
# 	labels.append(line.strip())
# file.close()
#
# y_true = []
# y_pred = []
# #load true and predict labels.
# file = open('predict.txt', 'r')
# lines = file.readlines()
# for line in lines:
# 	y_true.append(int(line.split(" ")[1].strip()))
# 	y_pred.append(int(line.split(" ")[0].strip()))
# file.close()
# tick_marks = np.array(range(len(labels))) + 0.5
#
# def plot_confusion_matrix(cm, title='Confusion Matrix', cmap = plt.cm.binary):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     xlocations = np.array(range(len(labels)))
#     plt.xticks(xlocations, labels, rotation=90)
#     plt.yticks(xlocations, labels)
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
# cm = confusion_matrix(y_true, y_pred)
# print(cm)
# np.set_printoptions(precision=2)
# cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
# print(cm_normalized)
# plt.figure(figsize=(12,8), dpi=120)
# #set the fontsize of label.
# #for label in plt.gca().xaxis.get_ticklabels():
# #    label.set_fontsize(8)
# #text portion
# ind_array = np.arange(len(labels))
# x, y = np.meshgrid(ind_array, ind_array)
#
# for x_val, y_val in zip(x.flatten(), y.flatten()):
#     c = cm_normalized[y_val][x_val]
#     if (c > 0.01):
#         plt.text(x_val, y_val, "%0.2f" %(c,), color='red', fontsize=7, va='center', ha='center')
# #offset the tick
# plt.gca().set_xticks(tick_marks, minor=True)
# plt.gca().set_yticks(tick_marks, minor=True)
# plt.gca().xaxis.set_ticks_position('none')
# plt.gca().yaxis.set_ticks_position('none')
# plt.grid(True, which='minor', linestyle='-')
# plt.gcf().subplots_adjust(bottom=0.15)
#
# plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# #show confusion matrix
# plt.show()