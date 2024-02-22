# Python 기본 라이브러리
import datetime
import os
import time
import argparse
import warnings

# 데이터 처리 및 시각화 관련 라이브러리
import matplotlib.pyplot as plt
import pandas as pd

# PyTorch 및 딥러닝 관련 라이브러리
import torch
from torch import nn
import torch.optim as optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# 사용자 정의 모듈
from data_utils import get_loader #커스텀파일
from models import CustomBlip2T5model #커스텀파일
from utils import create_unique_filename, save_loss_to_csv

# 기타 유틸리티
from tqdm import tqdm

# 경고 무시
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true" 


# argparser 설정
parser = argparse.ArgumentParser(description='학습 스크립트에 대한 인자 설정')
parser.add_argument('--total_step', type=int, default=30000, help='총 학습 스텝 수')
parser.add_argument('--train_batch_size', type=int, default=50, help='훈련 배치 크기')
parser.add_argument('--valid_batch_size', type=int, default=128, help='검증 배치 크기') 
parser.add_argument('--eval_every', type=int, default=500, help='몇 step마다 평가할건지') 
parser.add_argument('--pretrain_model_path', type=str, default='pretrain_model/blip2_pretrain1.pth', help='pretrain 모델 경로')

# 인자 파싱
args = parser.parse_args()
TOTAL_STEP = args.total_step # 학습 관련 상수 설정 
TRAIN_BATCH_SIZE = args.train_batch_size
VALID_BATCH_SIZE = args.valid_batch_size 
EVAL_EVERY = args.eval_every # 몇 step마다 평가할건지
pretrain_model_path = args.pretrain_model_path # 사전훈련된 모델 경로

print("***** train start *****")
print('TOTAL_STEP', TOTAL_STEP)  
print('TRAIN_BATCH_SIZE', TRAIN_BATCH_SIZE)
print('VALID_BATCH_SIZE', VALID_BATCH_SIZE) 
 
device = torch.device('cuda') 

# 출력 디렉터리 설정
output_dir_path = f'{os.getcwd()}/output'  
os.makedirs(output_dir_path, exist_ok=True) # 디렉터리가 없을 경우 생성

output_csv_path = f'{output_dir_path}/result.csv' 
output_csv_path = create_unique_filename(output_csv_path) # 이미 결과파일이 존재하면 이름을 변경  

# 모델 초기화 및 가중치 로드
blipT5_model = CustomBlip2T5model(img_size=560) 
if os.path.exists(pretrain_model_path): 
    pretrained_weight = torch.load(pretrain_model_path, map_location='cpu') # pretrain 가중치 삽입
    blipT5_model.load_state_dict(pretrained_weight['model'],strict=False)
    print('pretrain model loaded')
else: 
    print('pretrain model not loaded')
# 데이터 로더 준비
train_loader  = get_loader('train', TRAIN_BATCH_SIZE)
valid_loader  = get_loader('valid', VALID_BATCH_SIZE) 

# 검증 함수 정의
def validation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    
    start = time.time()
    model.eval()
    loss_sum = 0
    
    with torch.no_grad():  # Gradient 계산 방지
        for _, batch in enumerate(dataloader): 
            images = batch[0].to(device)
            questions = batch[1]
            answers = batch[2]  
            with torch.autocast(device_type='cuda'):
                outputs = model({"image": images,"text_input": questions, "text_output" : answers})           
            loss = outputs['loss']
            loss_sum += loss.item()    
    
    loss_mean = loss_sum / len(dataloader)
    end = time.time()
    sec = (end - start)  
    validation_time = str(datetime.timedelta(seconds=end - start)).split(".")[0]
    return loss_mean, validation_time 

# 옵티마이저 설정
optimizer = optim.AdamW(params=blipT5_model.parameters(), lr=1e-5)
# 스케줄러 설정 
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# 학습 시작 시간 기록 및 출력
start_time = datetime.datetime.now() 
print('\n',f'{str(start_time).split(".")[0]} 학습 시작')

# 학습 변수 초기화
global_step, minimum_loss = 0, float('inf')
 
# 학습 루프 
blipT5_model.to(device)
while global_step <= TOTAL_STEP:
    blipT5_model.train()
    total_batches = len(train_loader)
    for step, batch in enumerate(train_loader):
   
        images, questions, answers, scene_ID = batch
        images = images.to(device)
        with torch.autocast(device_type="cuda"):
            outputs = blipT5_model({"image": images,"text_input": questions, "text_output" : answers})
            loss = outputs['loss']  
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1 
        
        progress = global_step / total_batches * 100 
        
        # 손실 기록 및 저장
        if global_step % EVAL_EVERY == 0:
            valid_loss, validation_time = validation(blipT5_model, valid_loader, device)     
            print(f"global_step: {global_step}, Epoch Progress: {progress:.2f}%, train_loss: {loss.item():.4f}, val_loss: {valid_loss:.4f}, valid_time: {validation_time}")
            if valid_loss < minimum_loss: 
                minimum_loss = valid_loss 
                torch.save({
                        'step': global_step,
                        'model_state_dict': blipT5_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': valid_loss,
                        }, f"{output_dir_path}/model_best.pth")
                print(f"global_step: {global_step}, model saved")
         
        else:
            print(f"global_step: {global_step}, Epoch Progress: {progress:.2f}%, train_loss: {loss.item():.4f}") 
        
        save_loss_to_csv(output_csv_path, {
            'global_step': global_step,
            'train_loss': loss.item(),
            'val_loss': valid_loss if global_step % EVAL_EVERY == 0 else ''
            }) 
            
        
        if global_step > TOTAL_STEP:
                break
# 총 학습 시간 계산 및 출력
end = datetime.datetime.now()
print('총 학습 시간 : ',str(end - start_time))
print(f'{str(end).split(".")[0]} 학습 끝') 



# loss plot 저장
df = pd.read_csv(output_csv_path)
df['global_step'] = df['global_step'].astype(int)
df['train_loss'] = df['train_loss'].astype(float)
df['val_loss'] = df['val_loss'].astype(float) 

# 이동 평균을 위한 함수 정의
def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

# 이동 평균 적용
window_size = 5 # 이동 평균 윈도우 크기, 필요에 따라 조정
df['train_loss_ma'] = moving_average(df['train_loss'], window_size)

# 그래프 그리기
plt.plot(df['global_step'], df['train_loss_ma'], label='Train Loss')
df = df.dropna()
plt.plot(df['global_step'], df['val_loss'], label='Validation Loss')

# 그래프 제목 및 레이블 설정
plt.title('Smoothed Training and Validation Losses per Step')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()

# 그래프 이미지 파일로 저장
plt.savefig(output_csv_path.replace('csv','png'))
