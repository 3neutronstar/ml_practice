import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1개 채널 수의 이미지를 입력값으로 이용하여 6개 채널 수의 출력값을 계산하는 방식
        # Convolution 연산을 진행하는 커널(필터)의 크기는 3x3 을 이용
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Convolution 연산 결과 5x5 크기의 16 채널 수의 이미지
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device=device)

for j,p in enumerate(model.parameters()):
    for i,p_node in enumerate(p):
        if i%2==0:
            p[i]=torch.zeros_like(p_node)
            print(p_node,"Node")
    print(j,'layer',p)