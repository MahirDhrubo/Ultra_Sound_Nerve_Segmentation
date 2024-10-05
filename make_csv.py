import os
import csv
import pandas as pd 
import cv2
names = []
dir = 'train'
for f in os.listdir(dir):
    if os.path.isfile(os.path.join(dir,f)):
        temp = f.split('_')
        if len(temp) == 2:
            img = cv2.imread( os.path.join(dir,f))
            mask = cv2.imread(os.path.join(dir,f.split('.')[0]+'_mask.tif'))
            # print(img,mask)
            if img is not None and mask is not None:
                names.append(f)
#         else:
#             masks.append(f)
# print(len(names))
# print(len(masks))
# for i in range(len(names)):
#     p = names[i].split('_')
#     q = p[0]+'_'+p[1][:-4]+'_mask.tif'
#     print(names[i],masks.index(q))

df = pd.DataFrame(names)
df.to_csv('names.csv')
print(len(df))
# df = pd.read_csv('names.csv')
# print(df.iloc[2,1])
# img = cv2.imread(os.path.join('gapped_image',df.iloc[2,1]))
# print(img[0][0])