import os
path= './chapter3/dongsandcats/' # ��� ����
files=glob(os.path.join(path,'*/*.jpg')) #��� ������ ��� ���� �ҷ�����
print(f'Total no of images {len(files)}')

num_images=len(files)
shuffle=np.random.permutation(num_images) #������ ������ ����� �� ����� shuffle index ����

#������ image�� ������ ������ directory ����
os.mkdir(os.path.join(path,'valid')) 

#label name���� directory ����
for t in ['train','valid']:
    for folder in ['dog/','cat/']:
        os.mkdir(os.path.join(path,t,folder))

#valid dir�� image 2000�� ����
for i in shuffle[200:]:
    #���� �� ������ �ڵ�
    folder=files[i].split['/'][-1].split('.')[0]

