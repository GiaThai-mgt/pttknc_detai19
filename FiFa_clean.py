"""
FILE 1: TIỀN XỬ LÝ DỮ LIỆU & LỌC NGOẠI LAI ĐA BIẾN (MAHALANOBIS DISTANCE)
- Đề tài 19: Phân tích Thống kê Nhiều chiều (MANOVA 1 nhân tố trên dữ liệu FIFA 23)
- Chức năng:
    1. Đọc dữ liệu thô từ file CSV cố định 'archive/Fifa 23 Players Data.csv'.
    2. Loại bỏ khuyết thiếu (dropna), loại bỏ thủ môn (GK).
    3. Chuẩn hóa vị trí thi đấu thành 3 nhóm: Forward, Midfielder, Defender.
    4. Đổi tên 6 biến kỹ thuật sang tên ngắn gọn.
    5. Phát hiện & loại bỏ ngoại lai đa biến bằng khoảng cách Mahalanobis bình phương (ngưỡng Chi2 alpha=0.001, df=6).
    6. Thống kê và lưu số lượng mẫu bị loại ở từng bước vào 'results/cleaning_summary.csv'.
    7. Xuất tập dữ liệu sạch ra 'fifa_cleaned_no_outliers.csv'.
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

# C1: Chỉ lọc cảnh báo cụ thể (FutureWarning), không tắt cảnh báo toàn cục
warnings.filterwarnings('ignore', category=FutureWarning)

# BƯỚC 1: Dùng BASE_DIR để script chạy được từ bất kỳ thư mục làm việc nào và trên Google Colab
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "archive" / "Fifa 23 Players Data.csv"
OUTPUT_CLEAN_CSV = BASE_DIR / "fifa_cleaned_no_outliers.csv"
RESULTS_DIR = BASE_DIR / "results"

# ==============================================================================
# B5: BẢNG ÁNH XẠ TÊN CỘT GỐC SANG TÊN NGẮN GỌN
# ==============================================================================
# Bảng đối chiếu các biến kỹ thuật:
# + 'Passing Total'   -> 'Passing':   Chỉ số chuyền bóng tổng quát
# + 'Defending Total' -> 'Defending': Chỉ số kỹ năng phòng ngự tổng quát
# + 'Vision'          -> 'Vision':    Tầm nhìn chiến thuật và khả năng quan sát
# + 'Shooting Total'  -> 'Shooting':  Chỉ số dứt điểm và sút bóng tổng quát
# + 'Pace Total'      -> 'Pace':      Tốc độ bứt tốc và di chuyển
# + 'Stamina'         -> 'Stamina':   Thể lực và sức bền thi đấu
COLUMN_RENAME_MAP = {
    'Passing Total': 'Passing',
    'Defending Total': 'Defending',
    'Vision': 'Vision',
    'Shooting Total': 'Shooting',
    'Pace Total': 'Pace',
    'Stamina': 'Stamina'
}

# ==============================================================================
# B5: LÝ DO PHÂN NHÓM VỊ TRÍ THI ĐẤU
# ==============================================================================
# Trong bóng đá hiện đại và phân tích chiến thuật thể thao:
# 1. Nhóm Hậu vệ (Defender):
#    - Gồm các vị trí: CB (Trung vệ), LB/RB (Hậu vệ biên), LWB/RWB (Hậu vệ cánh dâng cao).
#    - Nhiệm vụ chủ yếu là tranh chấp, phòng ngự khu vực và bọc lót bảo vệ khung thành.
#
# 2. Nhóm Tiền vệ (Midfielder):
#    - Gồm các vị trí: CM (Tiền vệ trung tâm), CDM (Tiền vệ phòng ngự), CAM (Tiền vệ tấn công),
#      cùng với LM (Tiền vệ cánh trái) và RM (Tiền vệ cánh phải).
#    - LÝ DO GOM LM/RM VÀO NHÓM TIỀN VỆ (MIDFIELDER):
#      Trong sơ đồ chiến thuật (như 4-4-2, 4-2-3-1, 3-5-2), LM và RM đóng vai trò tiền vệ cánh
#      (wide midfielders), chịu trách nhiệm phân phối bóng, kiểm soát tuyến giữa dọc hành lang biên,
#      và tham gia hỗ trợ phòng ngự từ xa cùng các tiền vệ trung tâm, khác biệt với tiền đạo cánh thuần túy.
#
# 3. Nhóm Tiền đạo (Forward):
#    - Gồm các vị trí: ST (Tiền đạo cắm/Trung phong), CF (Hộ công/Tiền đạo lùi),
#      cùng với LW (Tiền đạo cánh trái) và RW (Tiền đạo cánh phải).
#    - LÝ DO GOM LW/RW VÀO NHÓM TIỀN ĐẠO (FORWARD):
#      LW và RW (wingers) trong bóng đá hiện đại (như sơ đồ 4-3-3, 3-4-3) chơi dâng cao sát vòng cấm
#      đối phương, ưu tiên nhiệm vụ đột phá, xâm nhập nách trung lộ và dứt điểm ghi bàn,
#      mang tính chất tấn công trực diện của tiền đạo (forwards).
def categorize_position(pos):
    pos = str(pos).strip().upper()
    if pos in ['ST', 'CF', 'LW', 'RW']:
        return 'Forward'
    elif pos in ['CM', 'CDM', 'CAM', 'LM', 'RM']:
        return 'Midfielder'
    elif pos in ['CB', 'LB', 'RB', 'LWB', 'RWB']:
        return 'Defender'
    return None


def clean_and_remove_outliers():
    """
    Thực hiện quy trình làm sạch dữ liệu, chuẩn hóa vị trí, loại bỏ ngoại lai Mahalanobis
    và lưu tóm tắt số lượng dòng bị loại vào thư mục results/.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GIAI ĐOẠN 1: TIỀN XỬ LÝ DỮ LIỆU BAN ĐẦU")
    print("=" * 80)

    # 1. Kiểm tra sự tồn tại của file CSV cố định
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại đường dẫn: {DATA_PATH}")

    print(f"Reading dataset from: {DATA_PATH}")

    # 2. Đọc dữ liệu ban đầu
    selected_cols = ['Best Position'] + list(COLUMN_RENAME_MAP.keys())
    df_raw = pd.read_csv(DATA_PATH, usecols=selected_cols)
    n_raw = len(df_raw)
    print(f"1. Tổng số dòng dữ liệu thô ban đầu: {n_raw:,}")

    # 3. Bước dropna: loại bỏ các dòng có giá trị khuyết thiếu (A5)
    df_dropna = df_raw.dropna().reset_index(drop=True)
    n_after_dropna = len(df_dropna)
    dropped_na = n_raw - n_after_dropna
    print(f"2. Loại bỏ giá trị khuyết thiếu (dropna): loại {dropped_na:,} dòng (còn lại {n_after_dropna:,} dòng)")

    # 4. Bước loại bỏ thủ môn 'GK' (A5)
    df_no_gk = df_dropna[df_dropna['Best Position'] != 'GK'].reset_index(drop=True)
    n_no_gk = len(df_no_gk)
    dropped_gk = n_after_dropna - n_no_gk
    print(f"3. Loại bỏ thủ môn (GK): loại {dropped_gk:,} dòng (còn lại {n_no_gk:,} dòng)")

    # 5. Bước chuẩn hóa vị trí thi đấu thành 3 nhóm (A5)
    df_no_gk['Position_Group'] = df_no_gk['Best Position'].apply(categorize_position)
    df_valid_pos = df_no_gk.dropna(subset=['Position_Group']).reset_index(drop=True)
    n_valid_pos = len(df_valid_pos)
    dropped_invalid_pos = n_no_gk - n_valid_pos
    print(f"4. Loại bỏ vị trí không thuộc 3 nhóm chính: loại {dropped_invalid_pos:,} dòng (còn lại {n_valid_pos:,} dòng)")

    # 6. Đổi tên các cột kỹ thuật
    df_valid_pos = df_valid_pos.rename(columns=COLUMN_RENAME_MAP)

    print("\n--- Phân bố số lượng mẫu theo từng nhóm (trước khi lọc ngoại lai): ---")
    print(df_valid_pos['Position_Group'].value_counts())

    # ==============================================================================
    # GIAI ĐOẠN 2: PHÁT HIỆN & LOẠI BỎ NGOẠI LAI ĐA BIẾN (MAHALANOBIS)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("GIAI ĐOẠN 2: PHÁT HIỆN & LOẠI BỎ NGOẠI LAI ĐA BIẾN (MAHALANOBIS)")
    print("=" * 80)

    features = list(COLUMN_RENAME_MAP.values())
    p = len(features)     # Số chiều p = 6
    alpha = 0.001         # Mức ý nghĩa alpha = 0.001

    X = df_valid_pos[features].values
    X_mean = np.mean(X, axis=0)
    S = np.cov(X, rowvar=False)

    # B2: Thay thế np.linalg.inv(S) bằng np.linalg.solve (tránh nghịch đảo trực tiếp)
    # diff = X - X_mean (N x p)
    # Giải hệ phương trình tuyến tính S * Y = diff^T => Y có kích thước p x N
    # Mahalanobis sq: D_M^2 = diff * S^-1 * diff^T = sum(diff * Y^T, axis=1)
    diff = X - X_mean
    try:
        y_solved = np.linalg.solve(S, diff.T).T
        mahalanobis_sq = np.sum(diff * y_solved, axis=1)
    except np.linalg.LinAlgError:
        # Fallback sử dụng pseudo-inverse nếu ma trận hiệp phương sai gần suy biến
        S_pinv = np.linalg.pinv(S)
        mahalanobis_sq = np.sum((diff @ S_pinv) * diff, axis=1)

    df_valid_pos['Mahalanobis_Distance_Sq'] = mahalanobis_sq

    # Ngưỡng tới hạn Chi-bình phương (p=6, alpha=0.001)
    threshold = chi2.ppf(1 - alpha, df=p)
    print(f"1. Số chiều p: {p}")
    print(f"2. Mức ý nghĩa alpha: {alpha}")
    print(f"3. Ngưỡng tới hạn Chi-bình phương (threshold): {threshold:.4f}")

    # Lọc bỏ ngoại lai
    outliers_mask = df_valid_pos['Mahalanobis_Distance_Sq'] > threshold
    num_outliers = int(outliers_mask.sum())
    df_clean = df_valid_pos[~outliers_mask].copy().reset_index(drop=True)
    n_clean = len(df_clean)

    print(f"4. Số lượng ngoại lai Mahalanobis bị loại: {num_outliers:,} dòng")
    print(f"5. Số dòng dữ liệu sạch còn lại: {n_clean:,} dòng, {df_clean.shape[1]} cột")

    # A5: Bảng tóm tắt số lượng dòng bị loại ở từng bước và lưu vào results/
    cleaning_summary = pd.DataFrame([
        {
            'Bước xử lý': '1. Dữ liệu thô ban đầu',
            'Số dòng còn lại': n_raw,
            'Số dòng bị loại': 0,
            'Tỷ lệ loại trên tổng (%)': 0.0
        },
        {
            'Bước xử lý': '2. Loại bỏ giá trị khuyết (dropna)',
            'Số dòng còn lại': n_after_dropna,
            'Số dòng bị loại': dropped_na,
            'Tỷ lệ loại trên tổng (%)': round(dropped_na / n_raw * 100, 2)
        },
        {
            'Bước xử lý': '3. Loại bỏ thủ môn (GK)',
            'Số dòng còn lại': n_no_gk,
            'Số dòng bị loại': dropped_gk,
            'Tỷ lệ loại trên tổng (%)': round(dropped_gk / n_raw * 100, 2)
        },
        {
            'Bước xử lý': '4. Loại vị trí ngoài 3 nhóm chính',
            'Số dòng còn lại': n_valid_pos,
            'Số dòng bị loại': dropped_invalid_pos,
            'Tỷ lệ loại trên tổng (%)': round(dropped_invalid_pos / n_raw * 100, 2)
        },
        {
            'Bước xử lý': '5. Lọc ngoại lai Mahalanobis (chi2, df=6, alpha=0.001)',
            'Số dòng còn lại': n_clean,
            'Số dòng bị loại': num_outliers,
            'Tỷ lệ loại trên tổng (%)': round(num_outliers / n_raw * 100, 2)
        }
    ])

    print("\n--- BẢNG TỔNG HỢP SỐ LƯỢNG MẪU BỊ LOẠI QUA TỪNG BƯỚC: ---")
    print(cleaning_summary.to_string(index=False))

    summary_file = RESULTS_DIR / "cleaning_summary.csv"
    cleaning_summary.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"\n=> Đã lưu bảng tổng hợp bước làm sạch vào: {summary_file}")

    # Xuất dữ liệu sạch ra file CSV
    df_clean.to_csv(OUTPUT_CLEAN_CSV, index=False, encoding='utf-8')
    print(f"=> Đã lưu tập dữ liệu sạch vào: {OUTPUT_CLEAN_CSV}")
    print("=" * 80)

    return df_clean


if __name__ == '__main__':
    # C2: Cấu hình encoding utf-8 khi chạy trực tiếp file
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    clean_and_remove_outliers()
