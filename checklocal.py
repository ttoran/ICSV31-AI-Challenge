import torch


# CUDA 사용 가능 여부 출력 (True이면 GPU 사용 가능)
print("CUDA 사용 가능 여부:", torch.cuda.is_available())

# 사용 가능한 GPU 개수 출력
print("사용 가능한 GPU 개수:", torch.cuda.device_count())

# 첫 번째 GPU 이름 출력 (GPU가 없으면 오류 발생할 수 있음)
if torch.cuda.is_available():
    print("첫 번째 GPU 이름:", torch.cuda.get_device_name(0))
