# Hướng dẫn chạy đề tài trên Visual Studio Code

## 1. Cài Python

Khuyến nghị Python **3.11.x**.

Kiểm tra:
```bash
python --version
```

Nếu Windows dùng `py`:
```bash
py --version
```

## 2. Mở thư mục dự án bằng VS Code

Mở VS Code -> File -> Open Folder -> chọn:
`Breast_Cancer_Wisconsin_VSCode`

Cấu trúc:
```text
Breast_Cancer_Wisconsin_VSCode/
├── main.py
├── requirements.txt
├── README.md
├── results/
└── plots/
```

## 3. Tạo môi trường ảo

Mở Terminal trong VS Code:

```powershell
py -3.11 -m venv .venv
```

Kích hoạt:

```powershell
.venv\Scripts\activate
```

Nếu PowerShell chặn script, có thể dùng Command Prompt trong VS Code và chạy:
```cmd
.venv\Scripts\activate.bat
```

Khi thành công sẽ thấy:
```text
(.venv)
```

## 4. Cài thư viện

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Không dùng:
```powershell
pip install -U numpy pandas scikit-learn ...
```
vì dễ làm xung đột phiên bản.

## 5. Chọn Python Interpreter

Trong VS Code:
- `Ctrl + Shift + P`
- gõ `Python: Select Interpreter`
- chọn `.venv\Scripts\python.exe`

## 6. Chạy toàn bộ chương trình

Trong Terminal:

```powershell
python main.py
```

Hoặc:
```powershell
py main.py
```

## 7. Kết quả

Sau khi chạy xong:

### results/
Các bảng CSV:
- `wdbc_prepared.csv`
- `baseline_results.csv`
- `feature_selection_results.csv`
- `mutual_information_scores.csv`
- `selected_features.csv`
- `feature_selected_model_results.csv`
- `baseline_vs_feature_selection.csv`
- `best_parameters.csv`
- `final_metrics.csv`
- `shap_feature_importance.csv`
- `test_predictions.csv`

### plots/
Các hình:
- `01_class_distribution.png`
- `02_correlation_heatmap.png`
- `03_baseline_roc.png`
- `04_feature_selection_k.png`
- `05_selected_features_mi.png`
- `06_final_confusion_matrix.png`
- `07_final_roc.png`
- `08_shap_summary.png`
- `09_shap_bar.png`
- `10_shap_waterfall_sample_1.png`

## 8. Ý nghĩa pipeline

```text
WDBC
  ↓
Kiểm tra dữ liệu
  ↓
Train/Test Split
  ↓
Baseline:
LR / KNN / SVM / Random Forest
  ↓
Mutual Information
  ↓
Chọn K đặc trưng tốt nhất
  ↓
Huấn luyện lại mô hình
  ↓
GridSearchCV
  ↓
SVM tối ưu
  ↓
Accuracy / Precision / Recall / Specificity / F1 / AUC
  ↓
SHAP XAI
```

## 9. Quy ước nhãn

Trong chương trình:
- `0 = Benign`
- `1 = Malignant`

Do đó `Recall/Sensitivity` đo khả năng phát hiện lớp **Malignant**.

## 10. Lưu ý về y khoa

Đây là mô hình nghiên cứu/giáo dục dựa trên WDBC. Không dùng kết quả để chẩn đoán bệnh nhân thực tế. WDBC không thay thế dữ liệu lâm sàng hoặc đánh giá độc lập trên bệnh viện.
## 11. Chạy ứng dụng web bằng Docker

Dự án sử dụng Docker để đóng gói ứng dụng Streamlit và các thư viện cần thiết trong môi trường độc lập.

### 11.1. Tạo Docker image

Mở Docker Desktop, sau đó mở PowerShell tại thư mục dự án và chạy:

```powershell
docker build -t breast-cancer-ai .
```

### 11.2. Khởi chạy ứng dụng

```powershell
docker run -d --name breast-cancer-app -p 8501:8501 breast-cancer-ai
```

Sau khi container khởi chạy, mở trình duyệt tại:

http://localhost:8501

### 11.3. Kiểm tra ứng dụng

Kiểm tra container đang chạy:

```powershell
docker ps
```

Xem nhật ký ứng dụng nếu có lỗi:

```powershell
docker logs breast-cancer-app
```

### 11.4. Dừng và khởi động lại

Dừng ứng dụng:

```powershell
docker stop breast-cancer-app
```

Khởi động lại:

```powershell
docker start breast-cancer-app
```

Nếu cần xóa container để tạo lại:

```powershell
docker rm -f breast-cancer-app
```

Sau đó chạy lại lệnh `docker run` ở mục 11.2.

## 12. Lưu ý về sử dụng

Ứng dụng phục vụ mục đích nghiên cứu và giáo dục trên bộ dữ liệu WDBC. Kết quả dự đoán không thay thế chẩn đoán của bác sĩ, không được dùng làm căn cứ duy nhất để quyết định điều trị và chưa thể hiện hiệu quả lâm sàng trên bệnh nhân thực tế.
