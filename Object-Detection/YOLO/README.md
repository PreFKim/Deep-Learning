# [YOLO](https://arxiv.org/abs/1506.02640)
본 페이지에서는 YOLO의 등장배경과 특징에 대해서 말하고자 합니다.

IoU와 Non Maximal Suppresion에 대해 잘 모르시는 분은 먼저 아래를 눌러 IoU와 Non Maximal Suppresion에 대해서 보고 와주세요

[IoU(Intersection over Union)](#51-iouintersection-of-union)

[Non Maximal Suppresion](#52-non-maximal-suppresion)


---

## 1. 기존의 모델들
기존의 Object Detection 방식들은 classifier를 detection 모듈로 재구성 해서 객체를 탐지했다.

대표적으로 DPM또는 R-CNN등이 있는데 특징은 다음과 같다.

- DPM은 전체 이미지에 대해서 Sliding window 방식을 사용해서 classifier를 사용하도록 한다.

- R-CNN은 먼저 Bounding box를 만들고 해당 bounding box에 대해서 classifier를 적용하는 방식이다.

이후에 후처리를 통해 중복된 bounding box와 같은 필요 없는 것을 제거하거나 다른 객체를 기반으로 box를 rescore하는 방식들이다.

이러한 구조 때문에 처리하는 속도가 느려지게 되고 각각에 요소에 대해서 학습하는 것이 어려워진다.

즉, object detection을 두 단계로 처리해서 이를 Two-stage detection이라고 한다.

## 2. YOLO

YOLO는 이와 다르게 Bounding box와 class에 대한 확률을 회귀 문제로 본다.

이 방법은 bounding box와 class에 대한 확률을 한번에 예측할 수 있게 한다.

이러한 구조 때문에 초당 45장 최대 155장의 이미지를 처리해 realtime detection이 가능하게 한다.

또한 기존의 realtime으로 객체 탐지가 가능한 SOTA모델들에 비해 더욱 더 정확한 예측을 할 수 있다.

YOLO의 구조는 간단하게 다음과 같이 보여줄 수 있다.

![1](./img/fig1.PNG)

이러한 YOLO의 장점은 세가지가 있다.

 1. YOLO는 매우 빠르고 객체 탐지를 회귀 문제로 해결해 복잡한 파이프라인이 필요없다.

 이로 인해 느리면 초당 45장의 이미지를 처리할 수 있고 빠르면 초당 150장의 이미지를 처리할 수 있다.

 2. YOLO는 예측을 할 때 이미지의 전체적인 부분에 대해서 근거를 둔다.

 sliding window 방식이나 일부분의 지역을 보는 기술과는 다르게 이미지의 전체적인 부분에 대해서 학습하기 때문에 context정보를 더 잘 얻는다.

 3. YOLO는 객체의 generalizable representation에 대해 학습한다.

 높은 generalize 때문에 새로운 상황이나 예측하지 못한 입력에 대해서도 높은 성능을 보인다.

## 2.1 Object detection Method

기존의 방법인 Two-stage 방식을 한번에 할 수 있게 되었다.

각각의 bounding box를 예측하기 위해 전체이미지로부터 feature들을 얻어내 사용했고 이 때문에 bounding box에 대해서 class를 예측할 수 있게 되었다.

그 방법은 다음과 같다.

1. 입력 이미지를 SxS의 격자로 나눈다.

2. 객체의 중심이 격자 cell에 있다면 그 격자 cell은 객체를 탐지한다.

3. 각각의 격자 cell은 B개의 Bounding box 정보와 C개의 class별 확률로 이루어져있다.

### Bounding box
각각의 bounding box는 (x,y,w,h,confidence)5개의 값을 가진다.

x,y는 해당 격자 셀의 경계로부터의 상대적인 좌표이다.

w,h는 전체 이미지로부터의 상대적인 너비와 높이이다.

### Confidence Score
confidence Score는 해당박스가 어떠한 객체를 포함하고 있는지에 대한 confidence이고 모델이 예측한 박스가 얼마나 정확한지를 나타낸다.

이를 공식화 하면 (객체에 대한 확률) * (실제와 예측의 box에 대한 IoU값)이다.

Cell내에 어떠한 객체도 없다면 confidence는 0이다.

confidence는 어떠한 ground truth 박스와 해당 박스의 iou 값이다.

### Class probablity
각각의 격자 cell은 C개의 클래스에 대한 확률 집합으로 구성되어 있다.Pr(클래스i|객체)

한개의 격자 cell당 하나의 클래스만을 예측한다.


### Finally
최종 test에서는 conditional class 확률과 각각의 box의 confidence를 곱한다.(식(1))

![2](./img/eqn1.PNG)

식(1)은 특정 클래스에 대한 각 box에 대한 confidence score를 제곤한다.

이러한 score는 box에 나타난 클래스에 대한 확률과 박스가 얼마나 객체에 맞게 예측이 되었는지를 나타낸다.

### Shape
본 논문에서는 Pascal VOC(20개의 클래스를 가짐)를 평가하기 위해

S=7 B=2 C=20을 사용해서 7x7x30 tensor를 가지게 되었다.

이때 출력의 모양은 SxSx(B*5+c) 이다.

이때 채널수가 B*5+C인 이유는 한개의 박스마다(x,y,w,h,confidence)를 갖고 하나의 격자 cell마다 C개의 클래스 확률을 갖기 때문이다.(격자당 하나의 Class만 예측하기때문이다.)

이때 하나의 격자당 최대 두개의 box만을 예측할 수 있기 때문에 하나의 이미지 내에서 7x7x2(98)개의 bounding box만을 예측할 수 있는 것이다.

![3](./img/fig2.PNG)

## 3. Network

## 3.1 Detail

Conv Layer는 feature들을 추출하고 반면에 FC Layer는 확률과 좌표들을 예측합니다.

본 논문의 구조는 GoogLeNet모델에 영향을 받았는데

24개의 Conv layer를 지난후 2개의 FC Layer와 연결된다.

이때 Inception module을 사용하는 것 대신에 1x1 reduction layer를 사용하며 3x3 conv layer를 사용한다.

그 구조는 아래 그림과 같다.

![4](./img/fig3.PNG)

YOLO는 Fast YOLO와 일반 YOLO가 있는데 앞에서 말한 모델은 일반 YOLO 모델이다.

이때 Fast YOLO모델은 24개의 레이어 대신 9개의 레이어를 사용한다.

모델의 크기는 다르지만 , 두 모델의 파라미터 수는 같다.


이때 입력 이미지는 fine grained된 특징을 더 잘 얻기 위해 224x224의 해상도를 448x448로 바꾸었다.

추가적으로 최종 활성함수를 제외한 나머지 함수를 Leaky ReLU함수를 사용하였고 아래와 같다.

![5](./img/eqn2.PNG)

## 3.2 Regression

YOLO 모델은 기존 모델들과 다르게 회귀문제로 Detection을 해내는데 이를 위해서 sum squared error를 optimize하는 방식을 사용한다.

이때 학습하기에는 더 쉬워지나, YOLO에 바로 쓰기에는 애매해진다.

그 이유는 다음과 같다.

1. Localization error(box에 대한 정보)를 Classification error(확률 분포)와 동일하게 가중치를 두는데 이는 이상적이지 않다(본 논문에서 30개의 채널에 대해서 모두 같은 수치라고 생각하기 때문이다.)

2. 모든 이미지에서 많은 격자 cell들은 어떤 객체도 포함하지 않는 경우가 생기는데 이러한 이유 때문에 confidence score가 0으로 향하게 학습이 된다.(이러한 점이 각각의 객체를 포함한 cell들의 gradient를 압도하는 경우가 생김[없다고 판단하게 만든다는 의미])

이를 해결하기 위해서 bounding box 좌표 예측의 loss를 증가시키고 객체를 포함하지 않은 박스들에 대한 confidence를 낮추기 위해 다음과 같은 파라미터를 정의했다.

λcoord = 5로 두고 λnoobj = 0.5로 설정했다.

Sum-squared erro는 또한 큰 box와 작은 box에서의 error 또한 동등하게 가중치를 두는데 

이때 YOLO의 error metric은 큰 box에서의 작은 편차가 작은 box에서보다 덜 중요하다는 것을 반영을 해야한다.

이를 어느정도 해결하기 위해서 bounding box의 너비와 폭에 square root를 적용한다.

## 3.3 Responsible
각 격자 cell에는 여러가지 bounding box가 있다.

이때 각 객체에 대해 하나의 bounding box predictor만을 원하는데

이때 Responsible 이라는 것은 여러 cell들에 존재하는 bounding box들 중에서 ground truth와의 IoU가 가장 높은 예측을 말한다.


## 3.4 Loss Function

![6](./img/eqn3.PNG)

이때 식에서 1(obj,i)는 cell i에서 객체가 나타남을 의미하고 1(obj,(i,j))는 i번째 cell에서 j번째 bounding box가 해당 예측에 대한 "responsible"이라는 것이다.

만약 격자 cell에서 객체가 보여지면 loss function은 classification error에만 영향을 받는다(따라서 앞에서 말한 조건부 확률과 같다)

또한 만약 predictor가 ground truth box에 대한 "responsible" 이라면 loss function은 오직 bounding box 좌표에만 영향을 받는다.

## 3.5 Inference
격자구조는 bounding box 예측에서 차원적인 측면으로 다양성을 강제하게 된다.

네트워크는 각각의 객체마다 하나의 객체만을 예측하는 것이 다양하지만 격자구조로 인해 하나의 객체에 여러 bounding box가 존재함을 의미한다.

Fig2를 보더라도 하나의 객체 주변에 대해 많은 bounding box가 존재한다.

이는 Non Maximal Suppresion을 통해 해결할 수 있다.

## 4. YOLO의 한계

본 논문에서의 YOLO 모델은 오직 두개의 box만을 예측하고 나머지는 클래스의 확률에 대해서 예측하는 구조 때문에 bounding box를 예측하는데에 큰 제약이 생긴다.

이런 제약은 한 이미지당 탐지할 수 있는 객체의 수를 제한한다.

앞에서 말했듯이 출력의 모양은 SxSx(B*5+c) 이라고 할때 한 셀당 bounding box에 대한 정보는 B개가 존재 하므로 한 이미지당 SxSxB개의 bounding box만 탐지할 수 있다.

본 논문의 YOLO 모델은 bounding box를 예측하기 위해 상대적으로 coarse feature를 사용하는데 이 이유는 입력이미지에 대해서 많은 downsampling layer를 가지기 때문이다.

또한 Loss function을 통해 학습을 하기는 하지만 큰 box에 대해서 작은 error는 널널하게 적용되지만 작은 box에서의 작은 error는 IoU에서 큰 영향을 미친다.

예를들어 4x4의 이미지 2x2 이미지 각각에서 의 한 픽셀 차이로 인한 IoU 값은 15/16이고 3/4로 작은 이미지에서의 IoU값 차이가 커지기 때문이다.


## 5. Others

## 5.1 IoU(Intersection over Union)

![7](./img/IoU.png)

IoU란 합집합에 대한 교집합의 비율이라고 보면 되는데 YOLO에서와 같이 두 박스에대한 IoU는 위의 그림과 같다고 보면 된다.

이때 두 박스가 거의 일치할 수록 교집합 부분이 많아져 IoU값이 높아진다.

IoU가 높을수록 두 박스가 일치하는 부분이 많다고 볼 수도 있다는 것이다.

Box A와 B의 중심좌표와 너비 높이를 아래와 같이 정의하면 아래와 같다.(x,y,w,h)

A = (Xa,Ya,Wa,Ha)
B = (Xb,Yb,Wb,Hb)

이때 두 Box의 사이의 X축거리 Y축 거리를 dx dy라고 한다.

합집합과 교집합 사이의 관계는 다음과 같다.
$$
Union(A,B) = A + B - Intersection(A,B)
$$

만약 두 박스가 겹친다고 할 때 Intersection(A,B)는 다음과 같다.

$$
Intersection(A,B) = dx \times dy
$$

이때 A와 B는 각각 다음과 같으므로

$$
A = Wa \times Ha 
$$

$$
B = Wb \times Hb 
$$



이를 이용해 IoU 를 수식으로 표현하면 다음과 같다.

$$
IoU = \frac{dx \times dy} {Wa \times Ha + Wb \times Hb + dx \times dy}
$$



## 5.2 Non Maximal Suppresion