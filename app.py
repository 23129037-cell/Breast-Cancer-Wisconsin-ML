
from pathlib import Path
import json
import os
import joblib
import pandas as pd
import streamlit as st

# =========================
# 1. CAU HINH GIAO DIEN
# =========================
st.set_page_config(
    page_title="Breast Cancer AI",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"

st.markdown("""
<style>
.stApp {
    background: #f5f8fc;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14213d, #243b64);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(115deg, #e9efff, #fce7f3);
    margin-bottom: 22px;
    border: 1px solid #e0e7ff;
}
.hero h1 {
    color: #172554;
    font-size: 32px;
    margin-bottom: 8px;
}
.hero p {
    color: #475569;
    font-size: 16px;
}
.metric-card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
}
.metric-label {
    color: #64748b;
    font-size: 14px;
}
.metric-value {
    color: #172554;
    font-size: 27px;
    font-weight: 700;
    margin-top: 5px;
}
.section-title {
    color: #172554;
    font-size: 22px;
    font-weight: 700;
    margin-top: 12px;
}
.note {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 13px;
    border-radius: 8px;
    color: #1e3a8a;
}
</style>
""", unsafe_allow_html=True)


# =========================
# 2. HAM DOC DU LIEU
# =========================
def read_csv(filename):
    path = RESULTS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def show_plot(filename, caption):
    path = PLOTS_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Chưa tìm thấy biểu đồ: {filename}")


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# 3. THANH DIEU HUONG
# =========================
st.sidebar.markdown("# 🎗️ Breast Cancer AI")
st.sidebar.caption("Machine Learning • Feature Selection • XAI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "ĐIỀU HƯỚNG",
    [
        "🏠 Tổng quan",
        "📊 So sánh mô hình",
        "🔬 Phân tích XAI",
        "🧪 Thử dự đoán",
        "🤖 AI Agent có xác nhận",
        "ℹ️ Giới thiệu"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Đồ án nghiên cứu • HCMUTE")


# =========================
# 4. BANNER
# =========================
st.markdown("""
<div class="hero">
    <h1>🎗️ HỆ THỐNG PHÂN TÍCH UNG THƯ VÚ</h1>
    <p>
        Ứng dụng học máy cổ điển kết hợp lựa chọn đặc trưng
        và trí tuệ nhân tạo có khả năng giải thích (XAI).
    </p>
</div>
""", unsafe_allow_html=True)


# =========================
# 5. TRANG TONG QUAN
# =========================
if page == "🏠 Tổng quan":
    st.markdown('<div class="section-title">Tổng quan hệ thống</div>',
                unsafe_allow_html=True)

    st.write(
        "Theo dõi dữ liệu, hiệu suất mô hình và các kết quả phân tích."
    )

    final_df = read_csv("final_metrics.csv")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Số mẫu dữ liệu", "569")
    with c2:
        metric_card("Số đặc trưng", "30")
    with c3:
        metric_card("Mô hình cuối", "SVM")
    with c4:
        if final_df is not None and "Accuracy" in final_df.columns:
            metric_card(
                "Accuracy",
                f"{final_df.iloc[0]['Accuracy']:.2%}"
            )
        else:
            metric_card("Accuracy", "N/A")

    st.markdown("### Phân bố dữ liệu")
    c1, c2 = st.columns(2)

    with c1:
        show_plot("01_class_distribution.png",
                  "Phân bố nhãn Benign và Malignant")

    with c2:
        show_plot("02_correlation_heatmap.png",
                  "Ma trận tương quan giữa các đặc trưng")

    st.markdown("### Hiệu suất mô hình cuối")
    if final_df is not None:
        display_cols = [
            col for col in [
                "Accuracy", "Precision", "Sensitivity_Recall",
                "Specificity", "F1", "ROC_AUC"
            ] if col in final_df.columns
        ]
        if display_cols:
            st.dataframe(
                final_df[display_cols].style.format("{:.2%}"),
                width="stretch",
                hide_index=True
            )
    else:
        st.warning("Chưa tìm thấy results/final_metrics.csv.")

    c1, c2 = st.columns(2)
    with c1:
        show_plot("06_final_confusion_matrix.png",
                  "Ma trận nhầm lẫn trên tập kiểm tra")
    with c2:
        show_plot("07_final_roc.png", "Đường cong ROC của mô hình cuối")

    st.markdown("""
    <div class="note">
    <b>Lưu ý:</b> Hệ thống phục vụ mục đích nghiên cứu và học tập.
    Kết quả dự đoán không thay thế chẩn đoán của bác sĩ.
    </div>
    """, unsafe_allow_html=True)


# =========================
# 6. SO SANH MO HINH
# =========================
elif page == "📊 So sánh mô hình":
    st.markdown("## So sánh hiệu suất mô hình")

    baseline = read_csv("baseline_results.csv")
    selected = read_csv("feature_selection_results.csv")

    if baseline is not None:
        st.markdown("### Trước lựa chọn đặc trưng")
        st.dataframe(baseline, width="stretch",
                     hide_index=True)

    if selected is not None:
        st.markdown("### Sau lựa chọn đặc trưng")
        st.dataframe(selected, width="stretch",
                     hide_index=True)

    show_plot("03_baseline_roc.png", "ROC các mô hình cơ sở")
    show_plot("04_feature_selection_k.png",
              "Lựa chọn số lượng đặc trưng")
    show_plot("05_selected_features_mi.png",
              "Điểm Mutual Information của đặc trưng")

    best = read_csv("best_parameters.csv")
    if best is not None:
        st.markdown("### Tham số mô hình tối ưu")
        st.dataframe(best, width="stretch",
                     hide_index=True)


# =========================
# 7. PHAN TICH XAI
# =========================
elif page == "🔬 Phân tích XAI":
    st.markdown("## Giải thích mô hình bằng SHAP")

    st.write(
        "SHAP giúp phân tích mức độ đóng góp của các đặc trưng "
        "đến đầu ra mô hình. Đây là giải thích mô hình, "
        "không phải bằng chứng về quan hệ nhân quả."
    )

    show_plot("08_shap_summary.png",
              "SHAP Summary: mức độ và chiều ảnh hưởng")
    show_plot("09_shap_bar.png",
              "Tầm quan trọng trung bình của đặc trưng")
    show_plot("10_shap_waterfall_sample_1.png",
              "Giải thích một mẫu kiểm tra")

    shap_df = read_csv("shap_feature_importance.csv")
    if shap_df is not None:
        st.markdown("### Bảng tầm quan trọng đặc trưng")
        st.dataframe(shap_df.head(15), width="stretch",
                     hide_index=True)


# =========================
# 8. THU DU DOAN
# =========================
elif page == "🧪 Thử dự đoán":
    st.markdown("## Thử dự đoán với mô hình SVM")
    st.warning(
        "Chỉ dùng để minh họa nghiên cứu. Không sử dụng kết quả "
        "này để tự chẩn đoán hoặc quyết định điều trị."
    )

    model_path = MODELS_DIR / "svm_model.joblib"
    names_path = MODELS_DIR / "feature_names.json"
    medians_path = MODELS_DIR / "feature_medians.json"

    if not model_path.exists():
        st.error(f"Không tìm thấy mô hình: {model_path}")
    elif not names_path.exists():
        st.error(f"Không tìm thấy danh sách đặc trưng: {names_path}")
    elif not medians_path.exists():
        st.error(f"Không tìm thấy giá trị trung vị: {medians_path}")
    else:
        model = joblib.load(model_path)

        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        with open(medians_path, "r", encoding="utf-8") as f:
            medians = json.load(f)

        st.info(
            "Các ô được điền sẵn bằng trung vị của tập huấn luyện. "
            "Bạn có thể thay đổi giá trị để thử nghiệm."
        )

        with st.form("prediction_form"):
            input_values = {}
            columns = st.columns(3)

            for i, feature in enumerate(feature_names):
                with columns[i % 3]:
                    input_values[feature] = st.number_input(
                        feature,
                        value=float(medians[feature]),
                        format="%.6f",
                        key=f"feature_{i}"
                    )

            submitted = st.form_submit_button(
                "🔍 Phân tích mẫu",
                width="stretch"
            )

        
        if submitted:
            if not confirmed:
                st.warning(
                    "Bạn cần đánh dấu xác nhận trước khi gọi API."
                )
            else:
                api_url = os.getenv(
                    "PREDICTION_API_URL",
                    "http://127.0.0.1:8000/predict"
                )

                try:
                    response = requests.post(
                        api_url,
                        json={"features": input_values},
                        timeout=30,
                    )

                    if response.status_code == 200:
                        result = response.json()

                        st.success("Dự đoán thành công!")
                        st.write("Kết quả:", result["label"])
                        st.write(
                            "Xác suất lành tính:",
                            f'{result["probability_benign"]:.2%}'
                        )
                        st.write(
                            "Xác suất ác tính:",
                            f'{result["probability_malignant"]:.2%}'
                        )
                        st.caption(result["notice"])
                    else:
                        st.error(
                            f"Lỗi API: {response.status_code}"
                        )
                        st.code(response.text)

                except requests.exceptions.RequestException as e:
                    st.error(f"Không thể kết nối API: {e}")

# =========================

# =========================
# 9. AI AGENT CO XAC NHAN
# =========================
elif page == "🤖 AI Agent có xác nhận":
    import requests

    st.markdown("## 🤖 Quy trình dự đoán có xác nhận")
    st.write(
        "Người dùng xem lại dữ liệu và xác nhận trước khi "
        "hệ thống gửi yêu cầu đến API dự đoán."
    )

    st.warning(
        "Chỉ phục vụ nghiên cứu và học tập; không sử dụng "
        "để tự chẩn đoán hoặc quyết định điều trị."
    )

    names_path = MODELS_DIR / "feature_names.json"
    medians_path = MODELS_DIR / "feature_medians.json"

    if not names_path.exists() or not medians_path.exists():
        st.error("Không tìm thấy file đặc trưng hoặc giá trị trung vị.")
    else:
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        with open(medians_path, "r", encoding="utf-8") as f:
            medians = json.load(f)

        st.info(
            "Các giá trị mặc định là trung vị của tập huấn luyện. "
            "Bạn có thể thay đổi để kiểm thử."
        )

        with st.form("agent_confirmation_form"):
            input_values = {}
            columns = st.columns(3)

            for i, feature in enumerate(feature_names):
                with columns[i % 3]:
                    input_values[feature] = st.number_input(
                        feature,
                        value=float(medians[feature]),
                        format="%.6f",
                        key=f"agent_feature_{i}",
                    )

            confirmed = st.checkbox(
                "Tôi đã xem lại dữ liệu và đồng ý gửi yêu cầu dự đoán.",
                key="agent_user_confirmation",
            )

            submitted = st.form_submit_button(
                "Xác nhận và gọi API",
                width="stretch",
            )

        if submitted:
            if not confirmed:
                st.warning(
                    "Bạn cần đánh dấu xác nhận trước khi gọi API."
                )
            else:
                api_url = os.getenv(
                    "PREDICTION_API_URL",
                    "http://127.0.0.1:8000/predict",
                )

                try:
                    response = requests.post(
                        api_url,
                        json={"features": input_values},
                        timeout=30,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success("Dự đoán thành công!")
                        st.write("Kết quả:", result["label"])
                        st.write(
                            "Xác suất lành tính:",
                            f'{result["probability_benign"]:.2%}',
                        )
                        st.write(
                            "Xác suất ác tính:",
                            f'{result["probability_malignant"]:.2%}',
                        )
                        st.caption(result["notice"])
                    else:
                        st.error(
                            f"Lỗi API: {response.status_code}"
                        )
                        st.code(response.text)

                except requests.exceptions.RequestException as e:
                    st.error(f"Không thể kết nối API: {e}")


# =========================
# 10. GIOI THIEU
# =========================
elif page == "ℹ️ Giới thiệu":
    st.markdown("## Giới thiệu đề tài")

    st.write("""
    **Mục tiêu:** Nghiên cứu khả năng phân loại dữ liệu ung thư vú
    bằng các thuật toán học máy cổ điển.

    **Dữ liệu:** Breast Cancer Wisconsin Diagnostic (WDBC),
    gồm 569 mẫu và 30 đặc trưng.

    **Các thuật toán:** Logistic Regression, KNN, SVM và Random Forest.

    **Lựa chọn đặc trưng:** Mutual Information kết hợp đánh giá
    hiệu suất bằng kiểm định chéo.

    **Tối ưu mô hình:** GridSearchCV.

    **Giải thích mô hình:** SHAP.

    Website này là công cụ minh họa kết quả nghiên cứu,
    không phải thiết bị y tế đã được thẩm định lâm sàng.
    """)
