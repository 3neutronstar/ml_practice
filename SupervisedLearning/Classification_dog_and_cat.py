import os
path= './chapter3/dongsandcats/' # ��� ����
files=glob(os.path.join(path,'*/*.jpg')) #��� ������ ��� ���� �ҷ�����
print(f'Total no of images {len(files)}')

num_images=len(files)
