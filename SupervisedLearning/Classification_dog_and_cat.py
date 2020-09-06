import os
path= './chapter3/dongsandcats/' # 경로 지정
files=glob(os.path.join(path,'*/*.jpg')) #경로 하위의 모든 파일 불러오기
print(f'Total no of images {len(files)}')

num_images=len(files)
