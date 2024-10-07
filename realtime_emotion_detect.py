import pyautogui
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deepface import DeepFace

def capture_and_detect():
    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # YOLOv8 모델 로드
    model = YOLO('yolov8n.pt')  # 'n'은 nano 모델을 의미합니다. 다른 크기의 모델을 사용할 수도 있습니다.
    model.to(device)

    while True:
        # 화면 캡처
        screenshot = pyautogui.screenshot()
        
        # PIL Image를 numpy 배열로 변환
        frame = np.array(screenshot)
        
        # RGB에서 BGR로 색상 순서 변경 (OpenCV는 BGR을 사용)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 이미지 크기 조정 (선택사항)
        scale_percent = 50  # 원본 크기의 50%
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
        # YOLOv8로 객체 탐지
        results = model(resized_frame, device=device)
        
        # 얼굴 감정 인식
        try:
            analysis = DeepFace.analyze(resized_frame, actions=['emotion'], enforce_detection=False, detector_backend='retinaface')
            for face in analysis:
                emotion = face['dominant_emotion']
                region = face['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                # 얼굴 주위에 박스 그리기
                cv2.rectangle(resized_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 감정 텍스트 추가
                cv2.putText(resized_frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error in emotion detection: {e}")

        # 탐지 결과 시각화
        annotated_frame = results[0].plot()
        
        # 캡처된 화면 표시
        cv2.imshow('Screen Capture with YOLO and Emotion', annotated_frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_detect()
