# 화면 캡처 및 객체 탐지 프로그램

이 프로그램은 실시간으로 화면을 캡처하고 YOLOv8을 사용하여 객체를 탐지합니다.

## 기능

- 실시간 화면 캡처
- YOLOv8을 사용한 객체 탐지
- GPU 가속 지원 (CUDA)
- 탐지 결과 실시간 시각화

## 설치 방법

1. 이 저장소를 클론합니다:
   ```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. 필요한 라이브러리를 설치합니다:
   ```
   pip install -r requirements.txt
   ```

3. CUDA가 설치되어 있는지 확인하세요. (GPU 사용 시)

## 사용 방법

1. 프로그램을 실행합니다:
   ```
   python realtime_window_detect.py
   ```

2. 프로그램이 실행되면 화면에 캡처된 이미지와 탐지된 객체가 표시됩니다.

3. 종료하려면 'q' 키를 누르세요.


## 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.