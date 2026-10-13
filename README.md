# TIỂU LUẬN PHÂN TÍCH THỐNG KÊ NHIỀU CHIỀU (ĐỀ TÀI 19)

**Đề tài:** Phân tích Phương sai Đa biến (MANOVA 1 nhân tố) so sánh 6 biến kỹ thuật (*Passing, Defending, Vision, Shooting, Pace, Stamina*) giữa 3 nhóm vị trí thi đấu (*Defender, Midfielder, Forward*) của cầu thủ bóng đá trong dữ liệu FIFA 23.

---

## 1. Yêu cầu môi trường & Cài đặt

* **Yêu cầu hệ thống:** Python >= 3.10
* **Tạo và kích hoạt môi trường ảo (Khuyên dùng):**
  * Trên Windows (PowerShell):
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
  * Trên Linux / macOS:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
* **Cài đặt các gói thư viện ghim phiên bản chính xác:**
  ```bash
  pip install -r requirements.txt
  ```

* **Hướng dẫn chạy trên Google Colab:**
  1. Nén toàn bộ thư mục `pttknc_detai19` thành file `.zip` và upload lên Google Colab hoặc Google Drive.
  2. Giải nén và chuyển vào thư mục dự án:
     ```bash
     !unzip pttknc_detai19.zip
     %cd pttknc_detai19
     ```
  3. Cài đặt các thư viện phụ thuộc:
     ```bash
     !pip install -r requirements.txt
     ```
  4. Thực thi lần lượt 2 bước:
     ```bash
     !python FiFa_clean.py
     !python fifa_analysis.py
     ```

---

## 2. Hướng dẫn chạy chương trình theo thứ tự

Chương trình được thiết kế chuẩn mực với `BASE_DIR = Path(__file__).resolve().parent`, cho phép bạn thực thi từ bất kỳ thư mục nào (thư mục hiện tại hoặc thư mục cha).

### Bước 1: Tiền xử lý & Lọc ngoại lai đa biến
```bash
python FiFa_clean.py
```
*(Nếu đứng từ thư mục cha: `python pttknc_detai19/FiFa_clean.py`)*

* **Chức năng:**
  * Đọc dữ liệu từ `archive/Fifa 23 Players Data.csv`.
  * Loại bỏ dữ liệu khuyết thiếu (`dropna`) và lọc bỏ thủ môn (`GK`).
  * Chuẩn hóa vị trí thi đấu thành 3 nhóm: `Defender`, `Midfielder`, `Forward`.
  * Tính khoảng cách Mahalanobis bình phương ($D_M^2$) trên không gian 6 chiều và loại bỏ ngoại lai đa biến theo ngưỡng Chi-bình phương ($\alpha = 0.001, df = 6$).
* **Kết quả đầu ra:**
  * File dữ liệu sạch: `fifa_cleaned_no_outliers.csv` ($16,386$ dòng).
  * Bảng thống kê số dòng bị loại qua từng bước: `results/cleaning_summary.csv`.

> **GHI CHÚ QUAN TRỌNG:** Nếu bạn muốn thực hiện làm sạch dữ liệu lại từ đầu (ví dụ thay đổi ngưỡng lọc ngoại lai), chỉ cần xóa file `fifa_cleaned_no_outliers.csv` rồi chạy lại lệnh trên.

---

### Bước 2: Phân tích thống kê đa biến, MANOVA & Xuất biểu đồ
```bash
python fifa_analysis.py
```
*(Nếu đứng từ thư mục cha: `python pttknc_detai19/fifa_analysis.py`)*

* **Chức năng:**
  * Đọc dữ liệu sạch `fifa_cleaned_no_outliers.csv` (nếu file chưa tồn tại, script sẽ tự động gọi `FiFa_clean.py` để xử lý).
  * Tính toán đặc trưng mẫu theo nhóm (Vector trung bình, Ma trận hiệp phương sai $S$, Ma trận tương quan $R$, Phương sai tổng quát $|S|$).
  * Kiểm định tính chuẩn đa biến Henze-Zirkler (**HZ Test**) theo từng nhóm.
  * Kiểm định tính đồng nhất ma trận hiệp phương sai (**Box's M Test**).
  * Thực thi mô hình **MANOVA 1 nhân tố** (Wilks' lambda, Pillai's trace, Hotelling-Lawley, Roy's largest root), tính giá trị tới hạn $F_{critical}$ ($\alpha = 0.05$) và quy tắc quyết định.
  * Thực hiện **ANOVA 1 chiều** cho từng biến, tính $\eta^2$ (eta-squared) và kiểm định so sánh cặp **Post-hoc Tukey HSD**.
  * Xuất 4 bộ biểu đồ độ phân giải cao (300 DPI) vào thư mục `output/`.
  * Ghi toàn bộ kết quả bảng biểu vào `results/` và log terminal vào `results/output.txt`.

---

## 3. Bảng ánh xạ tên cột gốc sang tên ngắn gọn

| Tên cột gốc trong CSV | Tên biến phân tích | Ý nghĩa chuyên môn |
|---|---|---|
| `Passing Total` | `Passing` | Chỉ số chuyền bóng tổng quát |
| `Defending Total` | `Defending` | Chỉ số kỹ năng phòng ngự tổng quát |
| `Vision` | `Vision` | Tầm nhìn chiến thuật và khả năng quan sát |
| `Shooting Total` | `Shooting` | Chỉ số dứt điểm và sút bóng tổng quát |
| `Pace Total` | `Pace` | Tốc độ bứt tốc và di chuyển |
| `Stamina` | `Stamina` | Thể lực và sức bền thi đấu |

---

## 4. Quy ước kiểm định Post-hoc Tukey HSD

Trong bảng kết quả `results/anova_posthoc.csv`:
* **Quy ước hiệu trung bình:** `Mean_Difference = Trung bình(Nhóm 2) - Trung bình(Nhóm 1)`.
  * Nếu `Mean_Difference > 0`: Nhóm 2 có chỉ số trung bình cao hơn Nhóm 1.
  * Nếu `Mean_Difference < 0`: Nhóm 2 có chỉ số trung bình thấp hơn Nhóm 1.
* **Quy ước hiển thị p-value (`p_display`):** Nếu giá trị $p < 0.001$, hiển thị dạng `< 0.001`; ngược lại làm tròn 4 chữ số thập phân.
* **Kết luận `Reject_H0 = True`:** Có sự khác biệt có ý nghĩa thống kê giữa hai nhóm ở mức ý nghĩa $\alpha = 0.05$.

---

## 5. Ý nghĩa các file đầu ra

### Thư mục hình ảnh: `output/` (300 DPI)
* `hinh_4_1_heatmap.png`: Ma trận tương quan Pearson toàn bộ mẫu và 3 ô con chi tiết theo từng nhóm vị trí (Defender, Midfielder, Forward).
* `hinh_4_2_scatter_ellipse.png`: Phân tách không gian Defending - Shooting kèm Elip tin cậy 95% và tọa độ tâm cụm.
* `hinh_4_3_pairplot.png`: Ma trận quan hệ cặp (Pairplot) giữa 6 biến kỹ thuật tô màu theo 3 nhóm vị trí ($n = 1,500$ mẫu ngẫu nhiên).
* `hinh_4_4_qq_mahalanobis.png`: Biểu đồ Q-Q plot của khoảng cách Mahalanobis bình phương ($D_M^2$) theo từng nhóm so với phân phối lý thuyết $\chi^2(df=6)$.

### Thư mục bảng số liệu & báo cáo: `results/`
* `cleaning_summary.csv`: Thống kê số lượng dòng dữ liệu bị loại ở từng bước tiền xử lý và tỷ lệ tương ứng.
* `mean_by_group.csv`: Bảng vector trung bình mẫu của 6 biến kỹ thuật theo 3 nhóm vị trí thi đấu.
* `cov_by_group.csv`: Ma trận hiệp phương sai mẫu $S$ cho từng nhóm vị trí.
* `corr_by_group.csv`: Ma trận hệ số tương quan mẫu $R$ cho từng nhóm vị trí.
* `hz_normality_by_group.csv`: Kết quả kiểm định tính chuẩn đa biến Henze-Zirkler theo từng nhóm vị trí kèm cột `p_display`.
* `box_m_test.csv`: Bảng thống kê kiểm định tính đồng nhất ma trận hiệp phương sai Box's M kèm cột `p_display`.
* `manova_results.csv`: Bảng tổng hợp 4 tiêu chuẩn kiểm định MANOVA kèm $F_{critical}$, `Decision_Rule` và `p_display`.
* `anova_posthoc.csv`: Kết quả ANOVA 1 chiều, $\eta^2$ và Post-hoc Tukey HSD so sánh từng cặp vị trí kèm `p_display`.
* `output.txt`: Toàn bộ nội dung hiển thị terminal trong suốt quá trình chạy `fifa_analysis.py`.
