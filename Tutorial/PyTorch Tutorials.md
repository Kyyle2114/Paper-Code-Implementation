![PyTorch_logo](./Images/PyTorch.png)

# PyTorch 

딥러닝을 위한 라이브러리, 파이토치 기초 사용법에 대한 포스팅입니다. 대부분의 내용은 [파이토치 한국어 튜토리얼](https://tutorials.pytorch.kr/)을 참고하였으니 더 많은 정보를 알기 위해서는 공식 튜토리얼을 참고해 주시기 바랍니다.

---

# Tensor

텐서(tensor)는 배열(array)이나 행렬(matrix)과 유사한 특수한 자료구조입니다. 텐서는 GPU나 다른 하드웨어 가속기에서 실행할 수 있다는 점만 제외하면 NumPy 의 ndarray와 유사합니다.

```python
import torch
import numpy as np

# tensor 직접 생성
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# numpy 배열로 생성
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 다른 tensor로 생성
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지합니다.
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씁니다.
print(f"Random Tensor: \n {x_rand} \n")
```
**출력**
```
Ones Tensor: 
 tensor([[1, 1],
        [1, 1]]) 

Random Tensor: 
 tensor([[0.0825, 0.6148],
        [0.2854, 0.4732]]) 
```


```python
# 무작위 또는 상수 값 사용하기
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```
**출력**
```
Random Tensor: 
 tensor([[0.7776, 0.3742, 0.6271],
        [0.1891, 0.6197, 0.6504]]) 

Ones Tensor: 
 tensor([[1., 1., 1.],
        [1., 1., 1.]]) 

Zeros Tensor: 
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```        
  
   
```python
# tensor의 속성 확인
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```
**출력**
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu

기본적으로 텐서는 CPU에 생성됩니다. to 메서드를 사용하면 (GPU의 가용성(availability)을 확인한 뒤) GPU로 텐서를 명시적으로 이동할 수 있습니다.

```python
# GPU가 존재하면 텐서를 이동
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

print(f"Device tensor is stored on: {tensor.device}")
```
**출력**
Device tensor is stored on: cuda:0


```python
# Numpy식 인덱싱과 슬라이싱 지원
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
```
**출력**
```
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```


```python
# tensor 합치기
t1 = torch.cat([tensor, tensor, tensor], dim=0)
print(t1)

t2 = torch.cat([tensor, tensor, tensor], dim=1)
print(t2)
```
**출력**
```
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

```python
# tensor 합치기
t3 = torch.stack([tensor, tensor, tensor], dim=0)
print(t3)

t4 = torch.stack([tensor, tensor, tensor], dim=1)
print(t4)
```
**출력**
```
tensor([[[1., 0., 1., 1.],
         [1., 0., 1., 1.],
         [1., 0., 1., 1.],
         [1., 0., 1., 1.]],

        [[1., 0., 1., 1.],
         [1., 0., 1., 1.],
         [1., 0., 1., 1.],
         [1., 0., 1., 1.]],

        [[1., 0., 1., 1.],
         [1., 0., 1., 1.],
         [1., 0., 1., 1.],
         [1., 0., 1., 1.]]])
tensor([[[1., 0., 1., 1.],
         [1., 0., 1., 1.],
         [1., 0., 1., 1.]],

        [[1., 0., 1., 1.],
         [1., 0., 1., 1.],
         [1., 0., 1., 1.]],

        [[1., 0., 1., 1.],
         [1., 0., 1., 1.],
         [1., 0., 1., 1.]],

        [[1., 0., 1., 1.],
         [1., 0., 1., 1.],
         [1., 0., 1., 1.]]])
```

```python
# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산. y1, y2, y3은 모두 같은 값
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# 요소별 곱(element-wise product)을 계산. z1, z2, z3는 모두 같은 값
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```
**출력**
```
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

```python
# tensor 집계
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```
**출력**
12.0 <class 'float'>


```python
# inplace 연산 - 원 객체를 변환
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
```
**출력**
```
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]]) 

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

- inplace 연산은 메모리를 일부 절약하지만, 기록(history)이 즉시 삭제되어 도함수(derivative) 계산에 문제가 발생할 수 있습니다. 따라서, 사용을 권장하지 않습니다.

```python
# tensor to numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```
**출력**
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]


```python
# inplace 연산은 넘파이에도 반영
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```
**출력**
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]


```python
# numpy to tensor
n = np.ones(5)
t = torch.from_numpy(n)

# inplace 연산은 tensor에도 반영
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```
**출력**
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]

# DATASET, DATALOADER

더 나은 가독성(readability)과 모듈성(modularity)을 위해 데이터셋 코드를 모델 학습 코드로부터 분리하는 것이 이상적입니다. PyTorch는 torch.utils.data.DataLoader 와 torch.utils.data.Dataset 의 두 가지 데이터 기본 요소를 제공하여 미리 준비해둔(pre-loaded) 데이터셋 뿐만 아니라 가지고 있는 데이터를 사용할 수 있도록 합니다. Dataset 은 샘플과 정답(label)을 저장하고, DataLoader 는 Dataset 을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 감쌉니다.

TorchVision 에서 Fashion-MNIST 데이터셋을 불러오는 예제를 살펴보겠습니다. Fashion-MNIST는 Zalando의 기사 이미지 데이터셋으로 60,000개의 학습 예제와 10,000개의 테스트 예제로 이루어져 있습니다. 각 예제는 흑백(grayscale)의 28x28 이미지와 10개 분류(class) 중 하나인 정답(label)으로 구성됩니다.

다음 매개변수들을 사용하여 FashionMNIST 데이터셋 을 불러옵니다.

- root 는 학습/테스트 데이터가 저장되는 경로입니다.
- train 은 학습용 또는 테스트용 데이터셋 여부를 지정합니다.
- download=True 는 root 에 데이터가 없는 경우 인터넷에서 다운로드합니다.
- transform 과 target_transform 은 특징(feature)과 정답(label) 변형(transform)을 지정합니다.

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# 데이터셋 인덱싱
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    # item: tensor -> number
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```
**출력**
![](https://velog.velcdn.com/images/kyyle/post/a5bcee56-9176-4b71-a727-e6a9c206ab90/image.png)

사용자 정의 Dataset 클래스는 반드시 3개 함수를 구현해야 합니다: \_\_init\_\_, \_\_len\_\_, and \_\_getitem\_\_.

아래 구현을 살펴보면 FashionMNIST 이미지들은 img_dir 디렉토리에 저장되고, 정답은 annotations_file csv 파일에 별도로 저장됩니다.

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

\_\_init\_\_ 함수는 Dataset 객체가 생성(instantiate)될 때 한 번만 실행됩니다. 여기서는 이미지와 주석 파일(annotation_file)이 포함된 디렉토리와 (다음 장에서 자세히 살펴볼) 두가지 변형(transform)을 초기화합니다.

labels.csv 파일은 다음과 같습니다.

```
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

\_\_len\_\_ 함수는 데이터셋의 샘플 개수를 반환합니다.

\_\_getitem\_\_ 함수는 주어진 인덱스 idx 에 해당하는 샘플을 데이터셋에서 불러오고 반환합니다. 인덱스를 기반으로, 디스크에서 이미지의 위치를 식별하고, read_image 를 사용하여 이미지를 텐서로 변환하고, self.img_labels 의 csv 데이터로부터 해당하는 정답(label)을 가져오고, (해당하는 경우) 변형(transform) 함수들을 호출한 뒤, 텐서 이미지와 라벨을 Python 사전(dict)형으로 반환합니다.

Dataset 은 데이터셋의 특징(feature)을 가져오고 하나의 샘플에 정답(label)을 지정하는 일을 한 번에 합니다. 모델을 학습할 때, 일반적으로 샘플들을 《미니배치(minibatch)》로 전달하고, 매 에포크(epoch)마다 데이터를 다시 섞어서 과적합(overfit)을 막고, Python의 multiprocessing 을 사용하여 데이터 검색 속도를 높이려고 합니다.

DataLoader 는 간단한 API로 이러한 복잡한 과정들을 추상화한 순회 가능한 객체(iterable)입니다.

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

DataLoader 에 데이터셋을 불러온 뒤에는 필요에 따라 데이터셋을 순회(iterate)할 수 있습니다. 아래의 각 순회(iteration)는 (각각 batch_size=64 의 특징(feature)과 정답(label)을 포함하는) train_features 와 train_labels 의 묶음(batch)을 반환합니다.

```python
# 이미지와 정답(label)을 표시합니다.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```
**출력**
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 4
![](https://velog.velcdn.com/images/kyyle/post/7ced0b85-0f0d-4884-bf4e-748f82765ac5/image.png)

# TRANSFORM

변형(transform) 을 해서 데이터를 조작하고 학습에 적합하게 만듭니다.

FashionMNIST 특징(feature)은 PIL Image 형식이며, 정답(label)은 정수(integer)입니다. 학습을 하려면 정규화(normalize)된 텐서 형태의 특징(feature)과 원-핫(one-hot)으로 부호화(encode)된 텐서 형태의 정답(label)이 필요합니다. 이러한 변형(transformation)을 하기 위해 ToTensor 와 Lambda 를 사용합니다.

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

ToTensor 는 PIL Image나 NumPy ndarray 를 FloatTensor 로 변환하고, 이미지의 픽셀의 크기(intensity) 값을 [0., 1.] 범위로 비례하여 조정(scale)합니다.

Lambda 변형은 사용자 정의 람다(lambda) 함수를 적용합니다. 여기에서는 정수를 원-핫으로 부호화된 텐서로 바꾸는 함수를 정의합니다. 이 함수는 먼저 (데이터셋 정답의 개수인) 크기 10짜리 영 텐서(zero tensor)를 만들고, scatter_ 를 호출하여 주어진 정답 y 에 해당하는 인덱스에 value=1 을 할당합니다.

# 신경망 모델 구성하기 

torch.nn 네임스페이스는 신경망을 구성하는데 필요한 모든 구성 요소를 제공합니다. PyTorch의 모든 모듈은 nn.Module 의 하위 클래스(subclass) 입니다. 신경망은 다른 모듈(계층; layer)로 구성된 모듈입니다. 이러한 중첩된 구조는 복잡한 아키텍처를 쉽게 구축하고 관리할 수 있습니다.

이어지는 장에서는 FashionMNIST 데이터셋의 이미지들을 분류하는 신경망을 구성해보겠습니다.

```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# device check
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```
**출력**
Using cuda device

```python
# 신경망 정의
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        
# instance 생성
model = NeuralNetwork().to(device)
print(model)        
```
**출력**
```
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

```python
# 입력 데이터 전달
X = torch.rand(1, 28, 28, device=device)

# model.forward()를 직접 호출 X
logits = model(X)

pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```
**출력**
Predicted class: tensor([1], device='cuda:0')

```python
# 28*28 이미지 3장
input_image = torch.rand(3,28,28)

# channel first
print(input_image.size())
```
**출력**
torch.Size([3, 28, 28])

```python
# nn.Flatten
flatten = nn.Flatten()
flat_image = flatten(input_image)

# dim=0의 미니배치 차원은 유지
print(flat_image.size())
```
**출력**
torch.Size([3, 784])

```python
# nn.Linear
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```
**출력**
torch.Size([3, 20])

```python
# nn.ReLU
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```
**출력**
```
Before ReLU: tensor([[-0.0169, -0.1533, -0.0035, -0.3462, -0.0321,  0.3911, -0.5633,  0.4065,
          0.3547,  0.6639,  0.2378, -0.0876,  1.0234, -0.1738, -0.3603, -0.7132,
          0.1323, -0.3048, -0.0770,  0.0808],
        [ 0.2363,  0.2990,  0.1733, -0.1744, -0.1985,  0.2695, -0.8865,  0.4389,
         -0.1790,  0.6451,  0.3115,  0.3208,  0.9010, -0.1441, -0.4344, -1.0550,
          0.2482, -0.3891, -0.0614,  0.2997],
        [-0.0418,  0.1123,  0.0650, -0.1968, -0.0783,  0.0950, -0.6734, -0.2330,
          0.1145,  0.3674, -0.0849,  0.3872,  0.3596, -0.0833, -0.3794, -0.7443,
          0.3674, -0.0749, -0.2195,  0.3969]], grad_fn=<AddmmBackward0>)


After ReLU: tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3911, 0.0000, 0.4065, 0.3547,
         0.6639, 0.2378, 0.0000, 1.0234, 0.0000, 0.0000, 0.0000, 0.1323, 0.0000,
         0.0000, 0.0808],
        [0.2363, 0.2990, 0.1733, 0.0000, 0.0000, 0.2695, 0.0000, 0.4389, 0.0000,
         0.6451, 0.3115, 0.3208, 0.9010, 0.0000, 0.0000, 0.0000, 0.2482, 0.0000,
         0.0000, 0.2997],
        [0.0000, 0.1123, 0.0650, 0.0000, 0.0000, 0.0950, 0.0000, 0.0000, 0.1145,
         0.3674, 0.0000, 0.3872, 0.3596, 0.0000, 0.0000, 0.0000, 0.3674, 0.0000,
         0.0000, 0.3969]], grad_fn=<ReluBackward0>)
```

```python
# nn.Sequential
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# nn.Softmax
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
pred_probab
```
**출력**
```
tensor([[0.1108, 0.0866, 0.1103, 0.0985, 0.0867, 0.1059, 0.1211, 0.0910, 0.0871,
         0.1020],
        [0.1182, 0.0851, 0.1144, 0.0936, 0.0802, 0.1068, 0.1197, 0.0929, 0.0887,
         0.1002],
        [0.1128, 0.0844, 0.1117, 0.0946, 0.0868, 0.1100, 0.1203, 0.0924, 0.0903,
         0.0967]], grad_fn=<SoftmaxBackward0>)
```

```python
# 모델의 parameters() 및 named_parameters() 메서드로 모든 매개변수에 접근 가능
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```
**출력**
```
Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)


Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0306, -0.0301,  0.0144,  ..., -0.0069,  0.0271,  0.0171],
        [ 0.0206,  0.0322,  0.0123,  ...,  0.0055, -0.0172, -0.0010]],
       device='cuda:0', grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([ 0.0134, -0.0334], device='cuda:0', grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[-0.0375, -0.0019,  0.0200,  ...,  0.0077,  0.0363,  0.0252],
        [ 0.0165,  0.0103,  0.0378,  ..., -0.0348,  0.0283,  0.0033]],
       device='cuda:0', grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([-0.0023,  0.0208], device='cuda:0', grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[ 0.0158,  0.0009, -0.0131,  ..., -0.0391,  0.0191, -0.0229],
        [ 0.0332,  0.0098, -0.0262,  ...,  0.0396, -0.0321,  0.0281]],
       device='cuda:0', grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([0.0198, 0.0383], device='cuda:0', grad_fn=<SliceBackward0>) 
```

# 모델 매개변수 최적화하기

이제 모델과 데이터가 준비되었으니, 데이터에 매개변수를 최적화하여 모델을 학습하고, 검증하고, 테스트할 차례입니다. 모델을 학습하는 과정은 반복적인 과정을 거칩니다.

각 반복 단계에서 모델은 출력을 추측하고, 추측과 정답 사이의 오류(손실(loss))를 계산하고,  매개변수에 대한 오류의 도함수(derivative)를 수집한 뒤, 경사하강법을 사용하여 이 파라미터들을 최적화(optimize)합니다.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
```

학습 시에는 다음과 같은 하이퍼파라미터를 정의합니다.

- 에포크(epoch) 수 - 데이터셋을 반복하는 횟수
- 배치 크기(batch size) - 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수
- 학습률(learning rate) - 각 배치/에포크에서 모델의 매개변수를 조절하는 비율. 값이 작을수록 학습 속도가 느려지고, 값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있습니다.

```
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

학습용 데이터를 제공하면, 학습되지 않은 신경망은 정답을 제공하지 않을 확률이 높습니다. 손실 함수(loss function)는 획득한 결과와 실제 값 사이의 틀린 정도(degree of dissimilarity)를 측정하며, 학습 중에 이 값을 최소화하려고 합니다. 주어진 데이터 샘플을 입력으로 계산한 예측과 정답(label)을 비교하여 손실(loss)을 계산합니다.

일반적인 손실함수에는 회귀 문제(regression task)에 사용하는 nn.MSELoss(평균 제곱 오차(MSE; Mean Square Error))나 분류(classification)에 사용하는 nn.NLLLoss (음의 로그 우도(Negative Log Likelihood)), 그리고 nn.LogSoftmax와 nn.NLLLoss를 합친 nn.CrossEntropyLoss 등이 있습니다.

모델의 출력 로짓(logit)을 nn.CrossEntropyLoss에 전달하여 로짓(logit)을 정규화하고 예측 오류를 계산합니다.

```python
# 손실 함수를 초기화합니다.
loss_fn = nn.CrossEntropyLoss()
```

학습하려는 모델의 매개변수와 학습률(learning rate) 하이퍼파라미터를 등록하여 옵티마이저를 초기화합니다.

```
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

학습 단계(loop)에서 최적화는 세단계로 이뤄집니다.
- optimizer.zero_grad()를 호출하여 모델 매개변수의 변화도를 재설정합니다. 기본적으로 변화도는 더해지기(add up) 때문에 중복 계산을 막기 위해 반복할 때마다 명시적으로 0으로 설정합니다.
- loss.backwards()를 호출하여 예측 손실(prediction loss)을 역전파합니다. PyTorch는 각 매개변수에 대한 손실의 변화도를 저장합니다.
- 변화도를 계산한 뒤에는 optimizer.step()을 호출하여 역전파 단계에서 수집된 변화도로 매개변수를 조정합니다.

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 예측(prediction)과 손실(loss) 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")    
```

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```
**출력**
```
Epoch 1
-------------------------------
loss: 2.294371  [   64/60000]
loss: 2.285928  [ 6464/60000]
loss: 2.265257  [12864/60000]
loss: 2.262064  [19264/60000]
loss: 2.237712  [25664/60000]
loss: 2.214598  [32064/60000]
loss: 2.226095  [38464/60000]
loss: 2.191674  [44864/60000]
loss: 2.183460  [51264/60000]
loss: 2.153970  [57664/60000]
Test Error: 
 Accuracy: 48.6%, Avg loss: 2.147559 
 
...
```

# 모델 저장하고 불러오기

이번 장에서는 저장하기나 불러오기를 통해 모델의 상태를 유지(persist)하고 모델의 예측을 실행하는 방법을 알아보겠습니다.

PyTorch 모델은 학습한 매개변수를 state_dict라고 불리는 내부 상태 사전(internal state dictionary)에 저장합니다. 이 상태 값들은 torch.save 메소드를 사용하여 저장(persist)할 수 있습니다.

```python
import torch
import torchvision.models as models

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

모델 가중치를 불러오기 위해서는, 먼저 동일한 모델의 인스턴스(instance)를 생성한 다음에 load_state_dict() 메소드를 사용하여 매개변수들을 불러옵니다.

```python
# 여기서는 weights 를 지정하지 않았으므로, 학습되지 않은 모델을 생성
model = models.vgg16() 
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

모델의 가중치를 불러올 때, 신경망의 구조를 정의하기 위해 모델 클래스를 먼저 생성(instantiate)해야 했습니다. 이 클래스의 구조를 모델과 함께 저장하고 싶으면, (model.state_dict()가 아닌) model 을 저장 함수에 전달합니다.

```python
# model save
torch.save(model, 'model.pth')

# model load
model = torch.load('model.pth')
```

---

이상으로 간단한 PyTorch 튜토리얼을 마치겠습니다. 더 자세한 정보는 [공식 문서](https://tutorials.pytorch.kr/)를 참고해 주세요.