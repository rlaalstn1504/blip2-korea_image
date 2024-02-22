import pandas as pd
from PIL import Image
import torch    
import pickle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler  
from lavis.processors import Blip2ImageTrainProcessor, BlipImageEvalProcessor

def load_image_from_pickle(pickle_path):
    """Pickle 파일로부터 이미지 데이터를 로드하고 PIL 이미지 형식으로 반환합니다."""
    with open(pickle_path, 'rb') as file:
        image_data = pickle.load(file)
    return image_data 

class Korean_gqa(torch.utils.data.Dataset):
    def __init__(self, mode):
        assert mode in ['train', 'valid', 'test'], f"mode must be one of ['train', 'valid', 'test']. Got {mode}"  # 조건에 맞지 않으면 에러 발생 
        if mode == 'train':
            self.vis_processors = Blip2ImageTrainProcessor(image_size=560) 
        else:
            self.vis_processors = BlipImageEvalProcessor(image_size=560)
        self.df = pd.read_csv(f'{mode}.csv')
                    
    def __getitem__(self, index):  
        ''' 
        따라서 미리 전처리를 한 후 pickle 파일로 만든 이미지를 읽는 방식으로 처리 중입니다. 
        '''
        
        # 전처리된 pickle 이미지가 없을 경우 이미지를 직접 로드하여 사용
        img  = Image.open(self.df['image_path'].iloc[index]).convert('RGB')
        img  = self.vis_processors(img)

        question =  self.df['question'].iloc[index]
        answer  =  self.df['answer'].iloc[index] 
        scene_ID =  self.df['Scene_Graph_ID'].iloc[index]
        return img, question, answer, scene_ID

    def __len__(self):
        return len(self.df)

 
def get_loader(mode, batch_size):
    dataset = Korean_gqa(mode)
    
    if mode == 'train': 
        dataset = Korean_gqa(mode) 
        data_sampler = RandomSampler(dataset) #RandomSampler : 랜덤
        batch_size = batch_size
        
    elif mode == 'valid':
        dataset = Korean_gqa(mode)
        data_sampler = SequentialSampler(dataset) #SequentialSampler : 항상 같은 순서
        batch_size = batch_size
    else:
        dataset = Korean_gqa(mode) 
        data_sampler = SequentialSampler(dataset) #SequentialSampler : 항상 같은 순서
        batch_size = batch_size

    
    data_loader = DataLoader(dataset,
                              sampler = data_sampler,
                              batch_size = batch_size,
                              num_workers = 4,
                              pin_memory=True, 
                              drop_last=False,
                              )
    return data_loader