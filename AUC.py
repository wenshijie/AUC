# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:02:23 2019

@author: lenovo
"""
import matplotlib.pyplot as plt
#计算样本中正例和负例的样本数
def read_data(file_path='H:/project/AUC/part-00000'):
    #[score,bool,bool] 输出概率和其真实标签，如果为为正则第二列为1如果为负样本则第三个为1否则为0
    samples = []
    pos, neg = 0, 0 #真实的正负样本数
    with open(file_path,'r') as f:
        for line in f:
            temp = eval(line)
            #score = '%.2f'% float(temp[0])
            score = float(temp[0])
            true_label = int(temp[1])
            temple = [score, 1, 0] if true_label == 1 else [score, 0, 1]
            pos += temple[1]
            neg += temple[2]
            samples.append(temple)
    return samples, pos , neg

#输出概率从大到小排序，并计算假阳率和真阳率
def sort_roc(samples, pos, neg):
    fp, tp = 0, 0 #假阳，真阳
    xy_fpr_tpr = []
    sample_sort = sorted(samples, key = lambda x:x[0], reverse=True)
    file=open('data.txt','w')
    file.write(str(sample_sort))
    file.close()
    for i in range(len(sample_sort)):
        fp += sample_sort[i][2]
        tp += sample_sort[i][1]
        xy_fpr_tpr.append([fp/neg, tp/pos])
    return xy_fpr_tpr
#画出ROC
def get_auc(xy_fpr_tpr):
    auc = 0.0
    pre_x = 0
    for x,y in xy_fpr_tpr:
        if x != pre_x:
            auc += (x-pre_x)*y
            pre_x = x
    return auc

def draw_roc(xy_fpr_tpr):
    x = [item[0] for item in xy_fpr_tpr]
    y = [item[1] for item in xy_fpr_tpr]
    plt.plot(x,y)
    plt.title('ROC curve (AUC = %.4f)' % get_auc(xy_fpr_tpr))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
            
if __name__ == '__main__':
    samples, pos, neg = read_data()
    xy = sort_roc(samples, pos, neg)
    draw_roc(xy)

        
   