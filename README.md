# NCKH_2023 - NGHIÊN CỨU TÌM HIỂU HỆ THỐNG PHÁT HIỆN HÀNH VI KHÔNG ĐỘI MŨ BẢO HIỂM KHI THAM GIA GIAO THÔNG ỨNG DỤNG TRÍ TUỆ NHÂN TẠO

Phương pháp được sử dụng: <br>

* Nhận diện vật thể (YOLOv8)
* Nhận dạng biển số xe (CNN)

## Install
```
pip install -r requirements.txt
```

## Usage
```
python main.py --source 'path_to_data' 
```

## Models

| Model            | size<br><sup>(pixels) | mAP<sup>val<br>50 | mAP<sup>val<br>50-95 | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------- | --------------------- | ----------------- | -------------------- | ------------------ | ----------------- |
| YOLOv8n-nckh2023 | 640                   | ~95%              | 59%                | 3.2                | 8.7               |
| YOLOv8n-nckh2023 | 640                   | ~97%              | 65,9%                | 25.9               | 78.9              |
