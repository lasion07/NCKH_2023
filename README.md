# NCKH_2023 - NGHIÊN CỨU TÌM HIỂU HỆ THỐNG PHÁT HIỆN HÀNH VI KHÔNG ĐỘI MŨ BẢO HIỂM KHI THAM GIA GIAO THÔNG ỨNG DỤNG TRÍ TUỆ NHÂN TẠO

**Phương pháp được sử dụng:**

* Nhận diện vật thể (YOLOv8)
* Nhận dạng biển số xe (CNN)

## Install
```
pip install -r requirements.txt
```
Tải mô hình nhận diện vật thể đặt trong folder "Models"

## Usage
```
python main.py --source 'path_to_data' 
```

## Models
**Mô hình nhận diện vật thể**
| Model            | size<br><sup>(pixels) | mAP<sup>val<br>50 | mAP<sup>val<br>50-95 | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------- | --------------------- | ----------------- | -------------------- | ------------------ | ----------------- |
| [YOLOv8n (nckh2023)](https://drive.google.com/drive/folders/13hkJmz5-yzaNbyhPb473kaYcFTA3f9nt?usp=sharing) | 640                   | ~95%              | 59%                | 3.2                | 8.7               |
| [YOLOv8m (nckh2023)](https://drive.google.com/drive/folders/13hkJmz5-yzaNbyhPb473kaYcFTA3f9nt?usp=sharing) | 640                   | ~97%              | 65,9%                | 25.9               | 78.9              |
 
**Mô hình nhận dạng ký tự**
[weight.h5](https://drive.google.com/drive/folders/13hkJmz5-yzaNbyhPb473kaYcFTA3f9nt?usp=sharing)
