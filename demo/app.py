import os
import sys
import glob
import gradio as gr
from pathlib import Path 

# PyTorch 및 딥러닝 관련 라이브러리
import torch
from PIL import Image 
from lavis.processors import BlipImageEvalProcessor

# streamlit 파일을 demo 폴더 안과 밖 모두 다 경로문제없이 실행하기 위해 path 추가
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.getcwd())

# 사용자 정의 모듈
from models import CustomBlip2T5model, remove_module_prefix  # 커스텀파일

def get_project_root() -> str:
    """Returns project root path."""
    return str(Path(__file__).parent)

def perform_visual_qa(image, question):
    """
    이미지와 질문을 기반으로 시각적 질문 답변(VQA)을 수행하는 함수
    """
    # 예시: 입력 이미지와 텍스트(질문)를 그대로 반환하는 더미 함수
    # 실제 모델을 사용할 경우 여기에 이미지 처리 및 모델 추론 코드를 추가
    return question  # 실제 사용 시 이 부분을 모델의 출력으로 대체

if __name__ == "__main__": 
    vis_processors = BlipImageEvalProcessor(image_size=560) # 이미지 전처리기 초기화
    blipT5_model = CustomBlip2T5model(img_size=560) # BLIP2-T5 모델 초기화

    best_state_dict = torch.load(f"{os.path.dirname(get_project_root())}/output/model_best.pth", map_location='cpu')['model_state_dict'] # 학습된 모델 가중치 불러오기 
    best_state_dict = remove_module_prefix(best_state_dict) # 모듈 접두사 제거
    blipT5_model.load_state_dict(best_state_dict,strict=False) # 학습된 모델 가중치 적용
    del best_state_dict # 메모리 효율을 위해 필요 없는 변수 삭제

    # 모델을 GPU로 이동
    device = 'cuda' 
    blipT5_model.to(device) 

    # VQA(Visual Question Answering) 테스트 함수
    def perform_visual_qa(image, question):
        # 이미지 불러오기 및 크기 조정
        image = Image.fromarray(image)
        # 이미지 전처리 및 모델 입력 형태로 변환
        image = vis_processors(image).unsqueeze(0).to(device) 
        
        # 모델을 이용한 답변 생성
        with torch.autocast(device_type="cuda"):
            output = blipT5_model.generate({"image": image, "prompt": question}) 
        return output[0]
    
    
    
    # 예시 이미지들이 위치한 경로
    examples_dir = f'{get_project_root()}/examples'
    examples_files = glob.glob(f'{examples_dir}/*')
    
    # 예시 이미지 파일 경로들과 질문의 리스트 생성
    # 여기서는 각 이미지에 대한 예시 질문을 "Example question for image?"로 설정합니다.
    # 실제 사용 시에는 각 이미지에 적절한 질문으로 대체해야 합니다.
    examples_list = [
        [img_path, "이 사진에 대해 간단히 설명해줘"] for img_path in examples_files
    ]

    # 이미지와 텍스트 입력을 받고, 텍스트 출력을 반환하는 Gradio 인터페이스 설정
    demo = gr.Interface(fn=perform_visual_qa, 
                        inputs=[gr.Image(), gr.Textbox(label="Question")], 
                        outputs=gr.Textbox(label="Answer"),
                        examples=examples_list,
                        examples_per_page=8)

    demo.launch(share=False)