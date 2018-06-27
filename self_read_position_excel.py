import numpy as np

import xlrd

def read_sample_position(excel_name,sheet_num,start_row,start_col,end_row,end_col) -> object:
    #excel_name = 'samples.xlsx'
    workbook = xlrd.open_workbook(excel_name)
    print('Reading locations of samples form excel:', excel_name)
    sheet_names= workbook.sheet_names()
    row_num = end_row - start_row + 1
    col_num = end_col - start_col + 1
    #print(row_num,col_num)
    #sample = np.empty([sheet_num,row_num,col_num],dtype=int)
    print('sheet_num in self_read_position_excel',sheet_num)
    #sample_temp = np.empty([14,2],dtype=int)
    #x = np.empty(14,dtype=int)
    #y = np.empty(2,dtype=int)
    #sheet_id = 0    #代表当前sheet的编号，0表示第一个
    #for sheet_name in sheet_names:
    sample_temp2=np.zeros((1,2))
    for sheet_id in range(sheet_num):
        sheet2 = workbook.sheet_by_index(sheet_id)#workbook.sheet_by_name(sheet_name)
        sample_temp = np.empty([row_num[sheet_id], col_num], dtype=int)
        for i in range(0,row_num[sheet_id]):
            for j in range(0,col_num):
                print(i,j,type(start_row))
                #sample_temp[i-1,j] = sheet2.cell(i, j).value
                if type(start_row) == list:
                    sample_temp[i, j] = sheet2.cell(i + start_row[sheet_id], j + start_col).value  #适用于起始行不同时
                else:
                    sample_temp[i, j] = sheet2.cell(i + start_row, j + start_col).value
            #sample[sheet_num,:,:] = sample_temp
        sheet_id += 1
        sample_temp2 = np.concatenate((sample_temp2,sample_temp),axis=0)
        print('sheet_id:',sheet_id,np.shape(sample_temp2))
    #print(sample[0,0,0],sample[0,0,1])
    sample = sample_temp2[1:,:]
    return sample
def read_sample_position_random(excel_name,sheet_num,num_per_class,total_per_class,start_col=1,start_row=1) -> object:
    #excel_name = 'samples.xlsx'
    class_num=sheet_num
    workbook = xlrd.open_workbook(excel_name)
    print('Reading locations of samples form excel:', excel_name)
    sheet_names= workbook.sheet_names()
    row_num=num_per_class
    col_num = 2
    index_per_class=[]
    for i in range(0, class_num):
        # index_per_class[i]=np.random.random(1,total_per_class[i],size=num_per_class[i])
        index_per_class.append(np.random.randint(1, total_per_class[i], size=num_per_class[i]))

    print('sheet_num in self_read_position_excel',sheet_num)

    sample_temp2=np.zeros((1,2))
    for sheet_id in range(sheet_num):
        sheet2 = workbook.sheet_by_index(sheet_id)#workbook.sheet_by_name(sheet_name)
        sample_temp = np.empty([row_num[sheet_id], col_num], dtype=int)
        for i in range(0,row_num[sheet_id]):
            for j in range(0,col_num):
                sample_temp[i, j] = sheet2.cell(index_per_class[sheet_id][i], j + start_col).value
            #sample[sheet_num,:,:] = sample_temp
        sheet_id += 1
        sample_temp2 = np.concatenate((sample_temp2,sample_temp),axis=0)
        print('sheet_id:',sheet_id,np.shape(sample_temp2))
    #print(sample[0,0,0],sample[0,0,1])
    sample = sample_temp2[1:,:]
    #print(sample)
    return sample

def read_sample_position_categoryID(excel_name,sheet_num,start_row,start_col,end_row,end_col) -> object:
    #excel_name = 'samples.xlsx'
    workbook = xlrd.open_workbook(excel_name)
    print('Reading locations of samples form excel:', excel_name)
    sheet_names= workbook.sheet_names()
    row_num = end_row - start_row + 1
    col_num = end_col - start_col + 1
    #print(row_num,col_num)
    #sample = np.empty([sheet_num,row_num,col_num],dtype=int)
    print('sheet_num in self_read_position_excel',sheet_num)
    #sample_temp = np.empty([14,2],dtype=int)
    #x = np.empty(14,dtype=int)
    #y = np.empty(2,dtype=int)
    #sheet_id = 0    #代表当前sheet的编号，0表示第一个
    #for sheet_name in sheet_names:
    #categoryID = np.zeros(1,1) #用于返回每一个sample的类别编号
    sample_temp2=np.zeros((1,3)) # 前两列分别存储行、列，第三列存储类别编号
    for sheet_id in range(sheet_num):
        sheet2 = workbook.sheet_by_index(sheet_id)#workbook.sheet_by_name(sheet_name)
        #categoryID_temp=np.empty(row_num[sheet_id])
        sample_temp = np.empty([row_num[sheet_id], col_num], dtype=int)
        for i in range(0,row_num[sheet_id]):
            for j in range(0,col_num):
                #print(i,j)
                #sample_temp[i-1,j] = sheet2.cell(i, j).value
                sample_temp[i, j] = sheet2.cell(i + start_row[sheet_id], j + start_col).value
            #sample[sheet_num,:,:] = sample_temp
            #categoryID_temp[i] = sheet_id

        sheet_id += 1
        #categoryID=np.concatenate((categoryID,categoryID_temp),axis=o)
        sample_temp2 = np.concatenate((sample_temp2,sample_temp),axis=0)  #数组拼接
        print('sheet_id:',sheet_id,np.shape(sample_temp2))
    #print(sample[0,0,0],sample[0,0,1])


    sample = sample_temp2[1:,:]
    return sample


if __name__ == '__main__':
    excel_name = 'samples.xlsx'
    read_sample_position_random(excel_name,4,[2,3,4,5],[60,60,60,60],1,1)