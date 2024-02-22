import os 
import csv
import pickle

def save_to_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def create_unique_filename(filename):
    base_path = '/'.join(filename.split('/')[:-1])
    filename = filename.split('/')[-1]
    # 파일 확장자와 이름을 분리
    name, ext = os.path.splitext(filename)

    # 파일의 초기 경로 설정
    full_path = os.path.join(base_path, filename)

    # 파일이 존재하는 경우 숫자를 증가시키며 새로운 파일명 생성
    count = 1
    while os.path.exists(full_path):
        new_filename = f"{name}{count}{ext}"
        full_path = os.path.join(base_path, new_filename)
        count += 1

    return full_path 


def save_loss_to_csv(output_csv_path,row_dict):
    fieldnames = list(row_dict.keys())
    with open(output_csv_path, newline="", mode="a") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if f.tell() == 0:
            writer.writeheader()

        writer.writerow(row_dict)