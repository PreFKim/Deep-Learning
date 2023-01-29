# Project

딥러닝을 활용해 프로젝트를 만들어 보았습니다.

## StopMotion

Segmentation을 활용해서 옷 부분만을 분리해낸 후 해당 마스크를 활용해 영상과 배경을 각각 multiply 연산을 해서 적절한 값들을 이용해 특정 배경에서 옷만 춤을 추는 영상으로 전환시킨다.

이때 영상을 구성하는 사진의 수와 프레임을 적절히 조절해서 스탑모션의 느낌을 내는 프로젝트이다.

## 화재 탐지 Segmentation

화재 이미지에 대해서 Segmentation 데이터 셋을 활용하여 학습한 후 웹캠을 사용하여 컴퓨터와 연결하여 실시간 화재 탐지 기능을 구현하였다.

## 치아 Segmentation

CBCT 3D 치아 이미지에 대해서 Segmentation 데이터 셋을 구축한 뒤 학습을 하여 해당 모델을 활용해 CBCT 3D 이미지 파일을 치아만을 분리해내서 3D STL 파일로 재구성한다.

이와 관련된 활동으로는 현재 UNET의 skip connection 구조를 변경해 성능을 UNET3+보다 향상시킨 내용과 함께 전자공학회 추계학술대회에서 내용에 대해서 발표 했고

이 내용을 토대로 현재 논문을 제출하여 심사 중에 있다.

## 해양 쓰레기 Classification

해양 쓰레기 Dataset에 대하여 Classification 을 진행한다.