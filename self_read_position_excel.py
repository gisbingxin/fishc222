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
                #print(i,j)
                #sample_temp[i-1,j] = sheet2.cell(i, j).value
                sample_temp[i, j] = sheet2.cell(i + start_row, j + start_col).value
            #sample[sheet_num,:,:] = sample_temp
        sheet_id += 1
        sample_temp2 = np.concatenate((sample_temp2,sample_temp),axis=0)
        print('sheet_id:',sheet_id,np.shape(sample_temp2))
    #print(sample[0,0,0],sample[0,0,1])
    sample = sample_temp2[1:,:]
    return sample

if __name__ == '__main__':
    read_sample_position()