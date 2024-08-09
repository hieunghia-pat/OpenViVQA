# Hướng dẫn Cài đặt và Chạy Dự án ViTextCaps Captioning

Mục tiêu của dự án là tạo chú thích cho hình ảnh một cách chính xác và hiệu quả.

## Hướng dẫn Cài đặt và Chạy

1. **Clone Repository**

   Đầu tiên, clone repository về máy của bạn bằng lệnh sau:

   ```bash
   !git clone -b NhiNguyen34-patch-1 https://github.com/NhiNguyen34/vitextcaps-captioning.git

2. Cài đặt các thư viện cần thiết:
    ```bash
    %cd vitextcaps-captioning 
    !pip install -r requirements.txt

3. Tạo thư mục và giải nén dữ liệu:
       ```bash

        !mkdir -p /content/vitextcaps-captioning/data/fasttext
   
        %cd /content/vitextcaps-captioning/data/fasttext
        !unzip /content/drive/MyDrive/ViTextCap/Data/newData/fasttext.zip
        
        %cd /content/vitextcaps-captioning/data
        !unzip /content/drive/MyDrive/ViTextCap/Data/features/vinvl_vinvl.zip
        !unzip /content/drive/MyDrive/ViTextCap/Data/features/swintextspotter.zip 

5. Chạy mô hình:
      ```bash
      
      !python /content/vitextcaps-captioning/train.py --config-file /content/vitextcaps-captioning/configs/mmf_m4c_captioner.yaml
