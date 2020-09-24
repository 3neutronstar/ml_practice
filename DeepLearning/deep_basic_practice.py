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

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

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
def cross_entropy_error1(y, t):
    if y.ndim == 1: #데이터가 하나인 경우에도 쓰일 수 있도록
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size #아직 이해 안됨!!!!!!!!!!!!

# 신경망 학습에서 지료플 정확도를 삼는다 => 매개변수의 사소한 변화에 거의 반응을 보이지 않음 // 반응이 있더라도 불연속적으로 갑자기 변화
# 이는 활성화 함수로 계단 함수보다 예를 들어 시그모이드 함수를 사용하는 이유와 비슷 -> 시그모이드 함수의 미분은 어느 장소라도 0이 되지 않음

# 모든 변수의 편미분을 벡터로 정리한 것 -> gradient
def numerical_gradient(f, x): # f는 편미분하고자 하는 함수 // x는 어느 점에서 알고싶은 값들의 list ex) [3.0, 4.0]
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        #f(x+h) 계산
        x[idx] = tmp_val +h
        fxh1 = f(x)

        #f(x-h) 계산
        x[idx] = tmp_val -h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad

#gradient descent
def gradient_descent(f, init_X, lr = 0.01, step_num = 100): # f : 최적화하려는 함수, init_x : 초기값, step_num : 경사법에 따른 반복 횟수
    x = init_X
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

print("-------------------------------------------------------")
#ex)
def funtion_2(x):
    return x[0] ** 2 + x[1] ** 2

init_X = np.array([-3.0, 4.0])
print(gradient_descent(funtion_2, init_X = init_X, lr = 0.1, step_num = 100))

# 간단한 신경망을 예를 들어 실제 기울기를 구하는 코드
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) #정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error1(y, t)

        return loss

#ex
net = simpleNet()
print(net.W) # 가중치 매개변수

x= np.array([0.6, 0.9])
p = net.predict(x)
print(p)
np.argmax(p)

t = np.array([0,0,1]) #정답레이블
print(net.loss(x, t))