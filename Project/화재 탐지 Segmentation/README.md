# 화재 탐지 Segmentation

화재를 조금 더 정밀하게 탐지하기 위해 Segmentation 모델을 사용했다.

비록 UNet 모델은 의료 이미지에 적합한 모델이지만 다른 segmentation 모델로도 충분히 segmentation이 가능하기에 우선 UNET 모델로 테스트 했다.

하이퍼 파라미터에 대한 얘기는 코드 내에 있습니다.

GTX 1060 그래픽 카드로 초당 10장의 이미지를 처리할 수 있다.

## 결과

### Test
![1](./result/test/1.jpg)

![2](./result/test/2.jpg)

![3](./result/test/3.jpg)

![4](./result/test/4.jpg)

### Train

![5](./result/train/1.jpg)

![6](./result/train/2.jpg)

![7](./result/train/3.jpg)