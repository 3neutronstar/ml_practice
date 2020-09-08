#손실함수 -> 신경망이 학습할 수 있도록 해주는 지표
#학습의 목표 -> 손실 함수의 겨로갓값을 가장 작게 만드는 가중지 매개변수를 찾는 것
#신경망의 특징 -> 데이터를 보고 학습할 수 있다는 점 (데이터를 보고 가중치 매개변수의 값을 자동으로 결정)
#손실함수 -> 오차제곱항과 교차엔트로피 오차를 사용

#오차 제곱항(결과값이 작을 수록 오차가 적다는 뜻)
import numpy as np
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)
t = [0,0,1,0,0,0,0,0,0,0] # 1이 있는 인덱스 즉 2가 정답이라고 가정
y = [0.1,0.05, 0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(sum_squares_error(np.array(y), np.array(t)))

#교차 엔트로피 오차 (결과값이 작을 수록 오차가 적다는 뜻)
def cross_entropy_error(y, t) :
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
t = [0,0,1,0,0,0,0,0,0,0] # 1이 있는 인덱스 즉 2가 정답이라고 가정
y = [0.1,0.05, 0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

# (배치용) 교차 엔트로피 오차 구현 (one-hot encoding 일 경우에 )
def cross_entropy_error(y, t):
    if y.ndim == 1: #데이터가 하나인 경우에도 쓰일 수 있도록
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# (배치용) 교차 엔트로피 오차 구현 (one-hot encoding이 아닐 경우에 )
def cross_entropy_error(y, t):
    if y.ndim == 1: #데이터가 하나인 경우에도 쓰일 수 있도록
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size #아직 이해 안됨!!!!!!!!!!!!

# 신경망 학습에서 지료플 정확도를 삼는다 => 매개변수의 사소한 변화에 거의 반응을 보이지 않음 // 반응이 있더라도 불연속적으로 갑자기 변화
# 이는 활성화 함수로 계단 함수보다 예를 들어 시그모이드 함수를 사용하는 이유와 비슷 -> 시그모이드 함수의 미분은 어느 장소라도 0이 되지 않음

