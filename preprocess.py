import pandas as pd
import os
import json
import pandas as pd 
import argparse 

parser = argparse.ArgumentParser(description='학습 스크립트에 대한 인자 설정')
parser.add_argument('--root_path', type=str, default='/data/2.최종산출물/1.데이터', help=' 1.Training, 2.Validation, 3.Test 폴더가 있는 경로 설정') 
parser.add_argument('--train_folder_name', type=str, default='1.Training', help=' 1.Training 폴더 이름 설정') 
parser.add_argument('--valid_folder_name', type=str, default='2.Validation', help=' 2.Validation 폴더 이름 설정')
parser.add_argument('--test_folder_name', type=str, default='3.Test', help=' 3.Test 폴더 이름 설정')

# 인자 파싱
args = parser.parse_args()
root_path = args.root_path # 학습 관련 상수 설정 
assert os.path.exists(root_path), f"1.Training, 2.Validation, 3.Test 폴더가 있는 경로를 rootpath 인자로 설정해주세요"

train_folder_name = args.train_folder_name 
valid_folder_name = args.valid_folder_name
test_folder_name = args.test_folder_name

train_path = f'{root_path}/{train_folder_name}/'
valid_path = f'{root_path}/{valid_folder_name}/'
test_path = f'{root_path}/{test_folder_name}/' 

scene_path = '2.라벨링데이터/LABEL/장면그래프/AI Hub 업로드/장면그래프.json'
qa_path = '2.라벨링데이터/LABEL/질의응답/AI Hub 업로드/질의응답.json'

def read_json_from_file(file_path):
    """
    주어진 파일 경로에서 JSON 파일을 읽고 파이썬 객체로 반환합니다.
    :param file_path: 읽을 파일의 경로
    :return: JSON 데이터를 포함한 파이썬 객체
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_and_format_scene_graph_data(raw_scene_path):
    """
    장면 그래프 데이터를 로드하고 메인 데이터프레임을 생성합니다.
    :param raw_scene_path: 장면 그래프 JSON 파일 경로
    :return: 생성된 메인 데이터프레임
    """
    data = read_json_from_file(raw_scene_path)
    df_main = pd.DataFrame(data)
    return df_main 
 
# 질의응답 데이터프레임으로 변환
def json_to_dataframe(json_data):
    # 데이터프레임을 만들기 위한 빈 리스트
    df_list = []

    # JSON 데이터를 순회하면서 필요한 정보 추출
    for qa_data in json_data:
        scene_id = qa_data['Scene_Graph_ID']
        for qa in qa_data['QA_list']:
            df_list.append({
                'sceneId': scene_id,
                'QA_ID': qa['QA_ID'],
                'question': qa['question'],
                'answer': qa['answer'],
                'question_type': ', '.join(qa['question_type']),
                'answer_type': qa['answer_type']
            })
    # 리스트를 데이터프레임으로 변환
    return pd.DataFrame(df_list) 

def load_and_preprocess_data(data_path, scene_path, qa_path):
    # 장면 그래프 데이터 로드 및 포맷
    scene_graph_df = load_and_format_scene_graph_data(data_path + scene_path)

    # 질의응답 데이터 로드 및 데이터프레임 변환
    qa_json = read_json_from_file(data_path + qa_path)
    df = json_to_dataframe(qa_json)
    df = df[['sceneId', 'QA_ID', 'question', 'answer', 'question_type', 'answer_type']]
    df.rename({'sceneId': 'Scene_Graph_ID'}, axis=1, inplace=True)
    df = df[df['answer_type'] == 'full_answer']

    return df, scene_graph_df

def merge_and_postprocess(df, scene_graph_df, data_path):
    # 메인 데이터프레임과 병합
    merged_df = pd.merge(df, scene_graph_df[['Scene_Graph_ID', 'Category']])
    merged_df = merged_df[['Scene_Graph_ID', 'Category', 'question', 'answer']]

    # 이미지 경로 추가
    merged_df['image_path'] = merged_df['Category'].apply(lambda x: data_path + '1.원천데이터/' + x)
    merged_df['image_path'] = merged_df['image_path'] + '/' + merged_df['Scene_Graph_ID'] + '.jpg'
    return merged_df 

try:
    # Train 데이터 처리
    train_df, scene_graph_df_train = load_and_preprocess_data(train_path, scene_path, qa_path)
    train_df = merge_and_postprocess(train_df, scene_graph_df_train, train_path) 
    print('train_df.shape', train_df.shape) 
    train_df.to_csv('train.csv') 
except: 
    print('train 전처리 실패. 경로 확인 요망')

try:
    # Validation 데이터 처리
    valid_df, scene_graph_df_valid = load_and_preprocess_data(valid_path, scene_path, qa_path)
    valid_df = merge_and_postprocess(valid_df, scene_graph_df_valid, valid_path) 
    valid_df.to_csv('valid.csv')
    print('valid_df.shape', valid_df.shape)
except:
    print('validation 전처리 실패. 경로 확인 요망')

try:
    # Test 데이터 처리
    test_df, scene_graph_df_test = load_and_preprocess_data(test_path, scene_path, qa_path)
    test_df = merge_and_postprocess(test_df, scene_graph_df_test, test_path) 
    print('test_df.shape', test_df.shape) 
    test_df.to_csv('test.csv')  
except: 
    print('test 전처리 실패. 경로 확인 요망')