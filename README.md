## 설치 방법

**필요한 패키지 설치**
   터미널을 열고 아래 명령어를 실행하여 PyTorch 및 관련 라이브러리를 설치합니다.

   ```bash
   pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

   cd ~/catkin_ws/src

   rm -rf ~/catkin_ws/src/*

   rm -rf .git*

   git clone https://github.com/jth5220/KIMM_ROS .

   cd ~/catkin_ws

   rosdep install --from-paths src --ignore-src -r -y
   
   catkin_make
   ```
  
**LaneNet 다운로드**
LaneNet 모델을 [여기](https://drive.google.com/file/d/1u-0ph3wNSyCTTW_1ODFSuoNeRv350hU_/view?usp=sharing)를 클릭하여 다운로드합니다.

파일 압축 해제 다운로드한 Ultrafast-Lane-Detection-Inference-Pytorch-.zip 파일의 압축을 해제합니다.

Ultrafast-Lane-Detection-Inference-Pytorch- 폴더를 /catkin/src/perception 디렉토리로 이동합니다.

**Clustering**
[여기](https://drive.google.com/drive/my-drive)를 클릭하여 다운로드합니다.
