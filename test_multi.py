import argparse
# argparser 설정
parser = argparse.ArgumentParser(description='학습 스크립트에 대한 인자 설정')
parser.add_argument('--test_batch_size', type=int, default=100, help='') 
parser.add_argument('--best_model_dir_path', type=str, default='output', help='')
parser.add_argument('--gpu_num', type=str, default='0,1,2,3', help='')

# 인자 파싱
args = parser.parse_args()
TEST_BATCH_SIZE = args.test_batch_size  
best_model_dir_path = args.best_model_dir_path
gpu_num = args.gpu_num 
print('gpu_num:', gpu_num)
import os 
os.environ["CUDA_VISIBLE_DEVICES"]= gpu_num 

# Python 기본 라이브러리

import glob
import random
import datetime
import time
import warnings

# 데이터 처리 및 시각화 관련 라이브러리
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# PyTorch 및 딥러닝 관련 라이브러리
import torch
from accelerate import Accelerator

# 사용자 정의 모듈
from data_utils import get_loader #커스텀파일
from models import CustomBlip2T5model, remove_module_prefix
from nltk.translate.bleu_score import sentence_bleu 
# 기타 유틸리티
from tqdm import tqdm
# 경고 무시
warnings.filterwarnings('ignore')
import torch.distributed as dist 

seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
torch.backends.cuda.matmul.allow_tf32 = False 
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = (
    False  # 이거 False해야 재현성이 유지되는 경우도 있음, True 시 자동으로 최적화 해줌
)
np.random.default_rng(seed)  # random 모듈의 Generator 클래스로 생성 시

start_time = datetime.datetime.now() 

blipT5_model = CustomBlip2T5model(img_size=560) 
test_loader  = get_loader('test', TEST_BATCH_SIZE)

best_state_dict = torch.load(f"{best_model_dir_path}/model_best.pth", map_location='cpu')['model_state_dict'] 
best_state_dict = remove_module_prefix(best_state_dict) 

blipT5_model.load_state_dict(best_state_dict,strict=False) 
del best_state_dict 

accelerator = Accelerator() 
if accelerator.is_main_process: 
    print(f'{str(start_time).split(".")[0]} Test 성능 측정 시작') 

blipT5_model, test_loader = accelerator.prepare(blipT5_model, test_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scene_ids_list = []
real_sentences_list = [] 
pred_sentences_list = []
bleu_scores = []  

# 각 GPU 프로세스에 대한 고유한 파일명 생성
process_index = accelerator.state.process_index
result_filename = f'test_result_{process_index}.csv' 

blipT5_model.eval()
for batch in tqdm(test_loader): 
    images = batch[0].to(device)
    questions = batch[1]
    answers = batch[2] 
    scene_ids = batch[3]
    outputs = blipT5_model.module.generate({"image": images,"prompt": questions}) 
    
    # 단어 토큰화: 한국어는 띄어쓰기를 기준으로 토큰화
    real_tokens = list(map(lambda x: x.split(),answers)) 
    pred_tokens = list(map(lambda x: x.split(),outputs))

    # 개별 BLEU 점수 계산
    for real_token, pred_token, scene_id in zip(real_tokens,pred_tokens,scene_ids):
        bleu_score = sentence_bleu([real_token], pred_token, weights=(1, 0, 0, 0))
        
        scene_ids_list.append(scene_id)
        real_sentences_list.append(' '.join(real_token))
        pred_sentences_list.append(' '.join(pred_token))
        bleu_scores.append(bleu_score)
        
        print(f"scene_ID:{scene_id}, 정답문장: {' '.join(real_token)}, 예측문장: {' '.join(pred_token)}, bleu-1_score: {round(bleu_score,2)}")


end_time = datetime.datetime.now() 


pd.DataFrame({'scene_id' : scene_ids_list,
            '정답문장' : real_sentences_list,
            '예측문장' : pred_sentences_list,
            '개별bleu_score' : bleu_scores}).to_csv(result_filename)  
 
if accelerator.is_main_process:
    time.sleep(10)
    # 모든 파일 병합
    all_results = pd.DataFrame()
    for part_filename in glob.glob('test_result_*.csv'):
        part_results = pd.read_csv(part_filename, index_col=0)
        all_results = pd.concat([all_results, part_results], ignore_index=True)
    
    # 최종 파일에 저장 
    all_results = all_results.drop_duplicates(['scene_id','정답문장'])
    all_results.to_csv('test_result.csv')

    print(f"{len(all_results)}개 문장에 대한 BLEU-1 score : {round(np.mean(all_results['개별bleu_score'])*100,2)}")

    sec = (end_time - start_time)
    print('총 소요 시간 : ',str(sec))
    print(f'{str(end_time).split(".")[0]} Test 성능 측정 끝') 
    
    df = pd.read_csv('test_result.csv', index_col = 0) 
    df = pd.concat([df,pd.DataFrame({'scene_id': (f'{len(bleu_scores)}개 문장에 대한 BLEU-1 score : {round(np.mean(bleu_scores)*100,2)}')},index=[0])]) 
    df.to_csv('test_result.csv')
    df.to_excel('test_result.xlsx') 
    
    # 임시 파일 삭제 (선택사항)
    for filename in glob.glob('test_result_*.csv'):
        os.remove(filename)