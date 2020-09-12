import os
path= './chapter3/dongsandcats/' # 경로 지정
files=glob(os.path.join(path,'*/*.jpg')) #경로 하위의 모든 파일 불러오기
print(f'Total no of images {len(files)}')

num_images=len(files)
shuffle=np.random.permutation(num_images) #데이터 집합을 만드는 데 사용할 shuffle index 생성

#검증된 image를 저장할 검증된 directory 생성
os.mkdir(os.path.join(path,'valid')) 

#label name으로 directory 생성
for t in ['train','valid']:
    for folder in ['dog/','cat/']:
        os.mkdir(os.path.join(path,t,folder))

#valid dir에 image 2000를 복사
for i in shuffle[200:]:
    #파일 명만 따오는 코드
    folder=files[i].split['/'][-1].split('.')[0]

