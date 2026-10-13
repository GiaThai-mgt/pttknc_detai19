"""
FILE 2: PHÂN TÍCH THỐNG KÊ ĐA BIẾN & TRỰC QUAN HÓA (MỤC 3.2, 3.3, 4.1, 4.2)
- Đề tài 19: Phân tích Thống kê Nhiều chiều (MANOVA 1 nhân tố trên dữ liệu FIFA 23)
- Chức năng:
    1. Đọc dữ liệu sạch 'fifa_cleaned_no_outliers.csv' (đã lọc ngoại lai từ FiFa_clean.py).
    2. Mục 3.2: Đặc trưng thống kê mẫu (Mean Vector, Covariance, Correlation, Generalized Variance) theo từng nhóm.
    3. Mục 3.2: Kiểm định giả định:
       - Kiểm định tính chuẩn đa biến Henze-Zirkler (HZ Test) theo từng nhóm vị trí.
       - Kiểm định tính đồng nhất ma trận hiệp phương sai Box's M Test kèm ghi chú chi tiết.
    4. Mục 3.3 & 4.1: Kiểm định MANOVA 1 nhân tố (Wilks' lambda, Pillai's trace, Hotelling-Lawley, Roy's root)
       kèm tính giá trị tới hạn F_critical (Mục 3.4) và quy tắc quyết định.
    5. ANOVA 1 chiều và Post-hoc Tukey HSD cho từng biến kỹ thuật (tính eta-squared, dùng thuộc tính công khai).
    6. Trực quan hóa dữ liệu xuất file ảnh chất lượng cao 300 DPI vào thư mục output/:
       - Hình 4.1: Ma trận tương quan tổng quát và theo từng nhóm (4 ô con).
       - Hình 4.2: Scatter plot phân tách Defending vs Shooting kèm Elip tin cậy 95%.
       - Hình 4.3: Pairplot 6 biến tô màu theo Position_Group (mẫu n = 1,500).
       - Hình 4.4: Q-Q plot của khoảng cách Mahalanobis bình phương so với phân phối Chi-square (df=6).
    7. Lưu toàn bộ các bảng kết quả CSV và log terminal vào thư mục results/ (thêm cột p_display).
"""

import sys
import warnings
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
import pingouin as pg
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns

# C1: Chỉ lọc cảnh báo cụ thể (FutureWarning, RuntimeWarning), không dùng ignore toàn cục
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# BƯỚC 1: Dùng BASE_DIR cho TẤT CẢ đường dẫn để chạy được từ bất kỳ đâu và trên Google Colab
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = BASE_DIR / "results"
cleaned_file = BASE_DIR / "fifa_cleaned_no_outliers.csv"

# Đảm bảo đường dẫn BASE_DIR có trong sys.path để import FiFa_clean
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# BƯỚC 5: Logger an toàn, chuyển tiếp thuộc tính sang terminal gốc qua __getattr__
class OutputLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.close()

    def __getattr__(self, name):
        return getattr(self.terminal, name)

# BƯỚC 4: Hàm định dạng hiển thị p-value: p < 0.001 ghi "< 0.001", ngược lại ghi 4 chữ số thập phân
def format_p_display(p_val):
    try:
        p_float = float(p_val)
        if p_float < 0.001:
            return "< 0.001"
        return f"{p_float:.4f}"
    except Exception:
        return str(p_val)

# C2: Thiết lập encoding utf-8 ở đầu file chính
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Khởi tạo logger ghi lại toàn bộ output terminal
logger = OutputLogger(RESULTS_DIR / "output.txt")
sys.stdout = logger

# Bọc toàn bộ chương trình chính trong try / finally để đảm bảo đóng log và khôi phục stdout (BƯỚC 5)
try:
    # B4: Cấu hình font chữ: ưu tiên 'DejaVu Sans' rồi mới đến 'Segoe UI', 'Arial'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 11

    # B1: Gọi sns.set_theme() DUY NHẤT một lần ở đầu file trước khi tạo bất kỳ figure nào
    sns.set_theme(style='white')

    # Danh sách 6 biến kỹ thuật và các nhóm vị trí
    features = ['Passing', 'Defending', 'Vision', 'Shooting', 'Pace', 'Stamina']
    groups = ['Defender', 'Midfielder', 'Forward']
    p = len(features)

    # Bảng màu nhất quán cho 3 nhóm vị trí thi đấu
    palette_colors = {
        'Defender': '#2563eb',    # Xanh dương đậm
        'Midfielder': '#10b981',  # Xanh lục ngọc
        'Forward': '#ef4444'      # Đỏ cam rực rỡ
    }

    # ==============================================================================
    # NẠP DỮ LIỆU SẠCH (ĐÃ LỌC NGOẠI LAI MAHALANOBIS TỪ FILE 1)
    # ==============================================================================
    if not cleaned_file.exists():
        print(f"Chưa tìm thấy '{cleaned_file}'. Đang tự động gọi module FiFa_clean.py để làm sạch...")
        try:
            import FiFa_clean
            df_clean = FiFa_clean.clean_and_remove_outliers()
        except Exception as e:
            raise FileNotFoundError(
                f"Vui lòng chạy file FiFa_clean.py trước để tạo '{cleaned_file}'. Lỗi: {e}"
            )
    else:
        print(f"Nạp dữ liệu sạch từ file: {cleaned_file}")
        df_clean = pd.read_csv(cleaned_file)

    print(f"Tổng số quan trắc: {len(df_clean):,} cầu thủ.")

    # ==============================================================================
    # GIAI ĐOẠN 3: TÍNH TOÁN ĐẶC TRƯNG MẪU & KIỂM ĐỊNH GIẢ ĐỊNH THỐNG KÊ (MỤC 3.2)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("MỤC 3.2: TÍNH TOÁN ĐẶC TRƯNG MẪU VÀ KIỂM ĐỊNH CÁC GIẢ ĐỊNH THỐNG KÊ")
    print("=" * 80)

    def calculate_multivariate_statistics(data, label):
        X_sub = data[features]
        mean_vec = X_sub.mean()
        cov_matrix = X_sub.cov()
        corr_matrix = X_sub.corr()
        gen_var = float(np.linalg.det(cov_matrix.values))
        
        print(f"\n>>> ĐẶC TRƯNG THỐNG KÊ MẪU - [{label}] (N = {len(data):,}) <<<")
        print("\n--- a. Vector trung bình mẫu (Mean Vector):")
        print(mean_vec.round(4).to_string())
        print("\n--- b. Ma trận hiệp phương sai mẫu S (Covariance Matrix):")
        print(cov_matrix.round(4))
        print("\n--- c. Ma trận tương quan mẫu R (Correlation Matrix):")
        print(corr_matrix.round(4))
        print(f"\n--- d. Phương sai tổng quát (Generalized Variance |S|): {gen_var:.4e}")
        
        return {
            'mean': mean_vec,
            'cov': cov_matrix,
            'corr': corr_matrix,
            'gen_var': gen_var
        }

    # Đặc trưng mẫu theo từng nhóm vị trí thi đấu
    stats_by_group = {}
    for g in groups:
        subset = df_clean[df_clean['Position_Group'] == g]
        stats_by_group[g] = calculate_multivariate_statistics(subset, label=f"NHÓM {g.upper()}")

    # A4: Lưu bảng Vector trung bình theo nhóm vào results/
    mean_summary_df = pd.DataFrame({g: stats_by_group[g]['mean'] for g in groups}).T
    mean_summary_df.index.name = 'Position_Group'
    mean_summary_df.round(4).to_csv(RESULTS_DIR / "mean_by_group.csv", encoding='utf-8')
    print(f"\n=> Đã lưu bảng vector trung bình theo nhóm vào: {RESULTS_DIR / 'mean_by_group.csv'}")

    # A4: Lưu ma trận hiệp phương sai S theo nhóm vào results/
    cov_list = []
    for g in groups:
        cov_g = stats_by_group[g]['cov'].copy()
        cov_g.insert(0, 'Position_Group', g)
        cov_list.append(cov_g)
    cov_combined_df = pd.concat(cov_list)
    cov_combined_df.round(4).to_csv(RESULTS_DIR / "cov_by_group.csv", encoding='utf-8')
    print(f"=> Đã lưu ma trận hiệp phương sai theo nhóm vào: {RESULTS_DIR / 'cov_by_group.csv'}")

    # A4: Lưu ma trận tương quan R theo nhóm vào results/
    corr_list = []
    for g in groups:
        corr_g = stats_by_group[g]['corr'].copy()
        corr_g.insert(0, 'Position_Group', g)
        corr_list.append(corr_g)
    corr_combined_df = pd.concat(corr_list)
    corr_combined_df.round(4).to_csv(RESULTS_DIR / "corr_by_group.csv", encoding='utf-8')
    print(f"=> Đã lưu ma trận tương quan theo nhóm vào: {RESULTS_DIR / 'corr_by_group.csv'}")

    # ==============================================================================
    # KIỂM ĐỊNH TÍNH CHUẨN ĐA BIẾN (HENZE-ZIRKLER TEST - C4: CHỈ THEO TỪNG NHÓM)
    # ==============================================================================
    print("\n" + "-" * 80)
    print(">>> KIỂM ĐỊNH TÍNH CHUẨN ĐA BIẾN THEO TỪNG NHÓM (HENZE-ZIRKLER TEST - alpha=0.05) <<<")
    print("-" * 80)

    hz_records = []
    for g in groups:
        subset = df_clean[df_clean['Position_Group'] == g][features]
        hz_grp = pg.multivariate_normality(subset, alpha=0.05)
        conclusion = 'Tuân theo chuẩn đa biến' if hz_grp.normal else 'KHÔNG tuân theo chuẩn đa biến'
        
        print(f"\n* Nhóm {g} (N = {len(subset):,}):")
        print(f"  - Giá trị thống kê HZ (HZ-Statistic): {hz_grp.hz:.4f}")
        print(f"  - Giá trị p-value: {hz_grp.pval:.4e} (Hiển thị: {format_p_display(hz_grp.pval)})")
        print(f"  - Có tuân theo chuẩn đa biến không (normal): {hz_grp.normal}")
        print(f"  - Kết luận: {conclusion}")
        
        hz_records.append({
            'Position_Group': g,
            'N': len(subset),
            'HZ_Statistic': round(float(hz_grp.hz), 4),
            'p_value': f"{hz_grp.pval:.4e}",
            'Normal': bool(hz_grp.normal),
            'Conclusion': conclusion,
            'p_display': format_p_display(hz_grp.pval)  # BƯỚC 4
        })

    # A4, B4: Lưu kết quả kiểm định HZ vào results/
    hz_df = pd.DataFrame(hz_records)
    hz_df.to_csv(RESULTS_DIR / "hz_normality_by_group.csv", index=False, encoding='utf-8')
    print(f"\n=> Đã lưu kết quả kiểm định HZ theo nhóm vào: {RESULTS_DIR / 'hz_normality_by_group.csv'}")

    # ==============================================================================
    # KIỂM ĐỊNH TÍNH ĐỒNG NHẤT CỦA MA TRẬN HIỆP PHƯƠNG SAI (BOX'S M TEST)
    # ==============================================================================
    print("\n" + "-" * 80)
    print(">>> KIỂM ĐỊNH TÍNH ĐỒNG NHẤT CỦA MA TRẬN HIỆP PHƯƠNG SAI (BOX'S M TEST) <<<")
    print("-" * 80)

    k = len(groups)
    n_list = [len(df_clean[df_clean['Position_Group'] == g]) for g in groups]
    df_list = [n - 1 for n in n_list]
    N_total = sum(n_list)
    df_total = sum(df_list)

    S_pooled = np.zeros((p, p))
    sum_df_ln_det = 0.0

    # C5: Xóa biến không dùng (sign_i, sign_pooled), chỉ lấy log-determinant
    for g, df_i in zip(groups, df_list):
        S_i = stats_by_group[g]['cov'].values
        S_pooled += df_i * S_i
        _, logdet_i = np.linalg.slogdet(S_i)
        sum_df_ln_det += df_i * logdet_i

    S_pooled /= df_total
    _, logdet_pooled = np.linalg.slogdet(S_pooled)

    M_stat = df_total * logdet_pooled - sum_df_ln_det
    c_correction = 1.0 - ((2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (k - 1))) * (sum(1.0 / df_i for df_i in df_list) - (1.0 / df_total))
    chi2_box_m = M_stat * c_correction
    df_box_m = int(p * (p + 1) * (k - 1) / 2)
    pval_box_m = chi2.sf(chi2_box_m, df=df_box_m)

    print("\n--- BẢNG THỐNG KÊ CHI TIẾT BOX'S M TEST:")
    print(f"  + Thống kê Box's M (M): {M_stat:.4f}")
    print(f"  + Hệ số hiệu chỉnh (c): {c_correction:.6f}")
    print(f"  + Thống kê Chi-square xấp xỉ (Chi2): {chi2_box_m:.4f}")
    print(f"  + Bậc tự do (df): {df_box_m}")
    print(f"  + p-value: {pval_box_m:.4e} (Hiển thị: {format_p_display(pval_box_m)})")
    print(f"  + Mức ý nghĩa alpha: 0.05")

    # A7: Nhánh if/else kèm ghi chú lý do nhạy cảm của Box's M với cỡ mẫu lớn và đề xuất dùng Pillai's trace
    if pval_box_m < 0.05:
        print("  + KẾT LUẬN: p-value < 0.05 => Bác bỏ giả thuyết H0.")
        print("    Ma trận hiệp phương sai giữa 3 nhóm vị trí thi đấu KHÔNG ĐỒNG NHẤT (Heteroscedasticity).")
        print("  + GHI CHÚ QUAN TRỌNG (A7):")
        print("    - Kiểm định Box's M cực kỳ nhạy cảm với kích thước mẫu lớn (ở đây n = 16,386 mẫu) và")
        print("      sự sai lệch nhỏ so với phân phối chuẩn đa biến, dẫn đến việc p-value hầu như luôn đạt ý nghĩa thống kê.")
        print("    - Do giả định đồng nhất ma trận hiệp phương sai bị vi phạm, trong kết quả phân tích MANOVA,")
        print("      chúng ta dựa chủ yếu vào tiêu chuẩn Pillai's Trace thay vì Wilks' Lambda.")
        print("      Pillai's Trace là tiêu chuẩn bền vững nhất (most robust) trước vi phạm giả định đồng nhất hiệp phương sai.")
    else:
        print("  + KẾT LUẬN: p-value >= 0.05 => Chưa đủ bằng chứng bác bỏ H0. Ma trận hiệp phương sai đồng nhất.")
        print("  + GHI CHÚ (A7):")
        print("    - Giả định đồng nhất ma trận hiệp phương sai được thỏa mãn. Có thể tin cậy sử dụng tiêu chuẩn Wilks' Lambda.")

    # A4, B4: Lưu kết quả Box's M vào results/ kèm cột p_display
    box_m_df = pd.DataFrame([
        {'Thống kê': "Thống kê Box's M (M)", 'Giá trị': f"{M_stat:.4f}", 'p_display': ''},
        {'Thống kê': "Hệ số hiệu chỉnh (c)", 'Giá trị': f"{c_correction:.6f}", 'p_display': ''},
        {'Thống kê': "Thống kê Chi-square xấp xỉ", 'Giá trị': f"{chi2_box_m:.4f}", 'p_display': ''},
        {'Thống kê': "Bậc tự do (df)", 'Giá trị': str(df_box_m), 'p_display': ''},
        {'Thống kê': "p-value", 'Giá trị': f"{pval_box_m:.4e}", 'p_display': format_p_display(pval_box_m)},
        {'Thống kê': "Mức ý nghĩa alpha", 'Giá trị': "0.05", 'p_display': ''},
        {'Thống kê': "Kết luận", 'Giá trị': "Bác bỏ H0 (Không đồng nhất hiệp phương sai)" if pval_box_m < 0.05 else "Chấp nhận H0 (Đồng nhất)", 'p_display': ''}
    ])
    box_m_df.to_csv(RESULTS_DIR / "box_m_test.csv", index=False, encoding='utf-8')
    print(f"\n=> Đã lưu kết quả kiểm định Box's M vào: {RESULTS_DIR / 'box_m_test.csv'}")

    # ==============================================================================
    # GIAI ĐOẠN 4: THỰC THI MÔ HÌNH MANOVA 1 NHÂN TỐ (MỤC 3.3, 3.4 & 4.1)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("MỤC 3.3, 3.4 & 4.1: THỰC THI MÔ HÌNH MANOVA 1 NHÂN TỐ")
    print("=" * 80)

    manova_formula = 'Passing + Defending + Vision + Shooting + Pace + Stamina ~ Position_Group'
    maov = MANOVA.from_formula(manova_formula, data=df_clean)
    manova_results = maov.mv_test()

    pos_group_summary = manova_results['Position_Group']['stat']
    print("\n--- BẢNG TỔNG HỢP 4 TIÊU CHUẨN KIỂM ĐỊNH MANOVA (Position_Group): ---")
    print(pos_group_summary.to_string())

    print("\n" + "-" * 80)
    print(">>> CHI TIẾT 4 TIÊU CHUẨN KIỂM ĐỊNH MANOVA CHO BÁO CÁO (alpha = 0.05) <<<")
    print("-" * 80)

    alpha_manova = 0.05
    manova_report_rows = []

    for stat_name in ["Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", "Roy's greatest root"]:
        val = float(pos_group_summary.loc[stat_name, 'Value'])
        f_stat = float(pos_group_summary.loc[stat_name, 'F Value'])
        num_df = float(pos_group_summary.loc[stat_name, 'Num DF'])
        den_df = float(pos_group_summary.loc[stat_name, 'Den DF'])
        pval = float(pos_group_summary.loc[stat_name, 'Pr > F'])
        
        # BƯỚC 2: Tính giá trị tới hạn F_critical = scipy.stats.f.ppf(0.95, Num_DF, Den_DF)
        f_critical = stats.f.ppf(0.95, num_df, den_df)
        conclusion = 'Bác bỏ H0' if pval < alpha_manova else 'Chấp nhận H0'
        
        print(f"\n* Tiêu chuẩn: {stat_name}")
        print(f"  - Giá trị kiểm định (Value): {val:.6f}")
        print(f"  - Thống kê F (F Value): {f_stat:.4f}")
        print(f"  - Bậc tự do tử số (Num DF): {int(num_df) if num_df.is_integer() else num_df}")
        print(f"  - Bậc tự do mẫu số (Den DF): {int(den_df) if den_df.is_integer() else den_df:.1f}")
        print(f"  - p-value (Pr > F): {pval:.4e} (Hiển thị: {format_p_display(pval)})")
        print(f"  - Giá trị tới hạn F (F Critical alpha=0.05): {f_critical:.4f}")
        print(f"  - Quy tắc quyết định (Decision Rule): F > F_critical => bác bỏ H0")
        print(f"  - Kết luận: {conclusion}")
        
        manova_report_rows.append({
            'Test_Statistic': stat_name,
            'Value': round(val, 6),
            'F_Value': round(f_stat, 4),
            'Num_DF': int(num_df) if num_df.is_integer() else num_df,
            'Den_DF': round(den_df, 1),
            'Pr_F': f"{pval:.4e}",
            'Conclusion': conclusion,
            'F_Critical_0.05': round(float(f_critical), 4),                      # BƯỚC 2
            'Decision_Rule': "F > F_critical => bác bỏ H0",                     # BƯỚC 2
            'p_display': format_p_display(pval)                                 # BƯỚC 4
        })

    # A4, B2, B4: Lưu bảng MANOVA 4 tiêu chuẩn vào results/
    manova_export_df = pd.DataFrame(manova_report_rows)
    manova_export_df.to_csv(RESULTS_DIR / "manova_results.csv", index=False, encoding='utf-8')
    print(f"\n=> Đã lưu kết quả 4 tiêu chuẩn MANOVA vào: {RESULTS_DIR / 'manova_results.csv'}")

    # ==============================================================================
    # A2: ANOVA 1 CHIỀU & POST-HOC TUKEY HSD CHO TỪNG BIẾN KỸ THUẬT
    # ==============================================================================
    print("\n" + "=" * 80)
    print("A2: ANOVA 1 CHIỀU VÀ POST-HOC TUKEY HSD CHO TỪNG BIẾN KỸ THUẬT")
    print("=" * 80)

    anova_posthoc_records = []

    for feat in features:
        # Lấy dữ liệu 3 nhóm cho biến feat
        sub_def = df_clean[df_clean['Position_Group'] == 'Defender'][feat].values
        sub_mid = df_clean[df_clean['Position_Group'] == 'Midfielder'][feat].values
        sub_fwd = df_clean[df_clean['Position_Group'] == 'Forward'][feat].values
        
        # 1. Tính ANOVA 1 chiều
        f_val, p_val = stats.f_oneway(sub_def, sub_mid, sub_fwd)
        
        # Tính Eta-squared (eta^2 = SS_between / SS_total)
        all_vals = df_clean[feat].values
        grand_mean = np.mean(all_vals)
        ss_between = (
            len(sub_def) * (np.mean(sub_def) - grand_mean)**2 +
            len(sub_mid) * (np.mean(sub_mid) - grand_mean)**2 +
            len(sub_fwd) * (np.mean(sub_fwd) - grand_mean)**2
        )
        ss_total = np.sum((all_vals - grand_mean)**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
        
        print(f"\n>>> BIẾN: {feat.upper()} <<<")
        print(f"  + ANOVA F-statistic: {f_val:.4f}, p-value: {p_val:.4e} (Hiển thị: {format_p_display(p_val)})")
        print(f"  + Eta-squared (eta^2): {eta_sq:.4f} ({eta_sq * 100:.2f}% phương sai được giải thích bởi nhóm vị trí)")
        
        # 2. Post-hoc Tukey HSD bằng statsmodels
        tukey = pairwise_tukeyhsd(endog=df_clean[feat], groups=df_clean['Position_Group'], alpha=0.05)
        print("  + Kết quả Post-hoc Tukey HSD:")
        print(tukey.summary())
        
        # BƯỚC 3: Thay tukey._results_table.data bằng các thuộc tính công khai
        # Ghép cặp bằng itertools.combinations(tukey.groupsunique, 2) theo đúng thứ tự statsmodels
        # Chú thích: Mean_Difference = trung bình nhóm 2 trừ nhóm 1.
        pair_list = list(itertools.combinations(tukey.groupsunique, 2))
        for idx, (g1, g2) in enumerate(pair_list):
            meandiff = tukey.meandiffs[idx]
            p_adj = tukey.pvalues[idx]
            lower_ci, upper_ci = tukey.confint[idx]
            reject = tukey.reject[idx]
            
            anova_posthoc_records.append({
                'Variable': feat,
                'ANOVA_F': round(float(f_val), 4),
                'ANOVA_p_value': f"{p_val:.4e}",
                'Eta_Squared': round(float(eta_sq), 4),
                'Group_1': g1,
                'Group_2': g2,
                'Mean_Difference': round(float(meandiff), 4),
                'p_adj': f"{round(float(p_adj), 4):.4e}",
                'CI_Lower_95': round(float(lower_ci), 4),
                'CI_Upper_95': round(float(upper_ci), 4),
                'Reject_H0': bool(reject),
                'p_display': format_p_display(p_adj)                           # BƯỚC 4
            })

    # A2, B4: Lưu kết quả ANOVA và Tukey HSD vào results/anova_posthoc.csv
    anova_posthoc_df = pd.DataFrame(anova_posthoc_records)
    anova_posthoc_df.to_csv(RESULTS_DIR / "anova_posthoc.csv", index=False, encoding='utf-8')
    print(f"\n=> Đã lưu thành công kết quả ANOVA và Tukey HSD vào: {RESULTS_DIR / 'anova_posthoc.csv'}")

    # ==============================================================================
    # GIAI ĐOẠN 5: TRỰC QUAN HÓA DỮ LIỆU ĐA BIẾN XUẤT ẢNH BÁO CÁO (MỤC 4.2)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("MỤC 4.2: TRỰC QUAN HÓA DỮ LIỆU ĐA BIẾN (XUẤT FILE ẢNH DPI 300 VÀO output/)")
    print("=" * 80)

    # ------------------------------------------------------------------------------
    # Biểu đồ 1 (B6): Ma trận tương quan toàn bộ và 3 ô con theo từng nhóm vị trí
    # ------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=300)

    # (a) Toàn bộ dữ liệu
    corr_overall = df_clean[features].corr()
    sns.heatmap(
        corr_overall, annot=True, fmt='.2f', cmap='coolwarm',
        vmin=-1.0, vmax=1.0, ax=axes[0, 0],
        linewidths=0.8, linecolor='white',
        annot_kws={'size': 10, 'weight': 'bold'},
        cbar_kws={'label': 'Hệ số r'}
    )
    axes[0, 0].set_title(f"(a) Toàn bộ tập dữ liệu (N = {len(df_clean):,})", fontsize=12.5, weight='bold', pad=12)

    # (b) Nhóm Defender
    corr_def = df_clean[df_clean['Position_Group'] == 'Defender'][features].corr()
    sns.heatmap(
        corr_def, annot=True, fmt='.2f', cmap='coolwarm',
        vmin=-1.0, vmax=1.0, ax=axes[0, 1],
        linewidths=0.8, linecolor='white',
        annot_kws={'size': 10, 'weight': 'bold'},
        cbar_kws={'label': 'Hệ số r'}
    )
    axes[0, 1].set_title(f"(b) Nhóm Hậu vệ - Defender (n = {len(df_clean[df_clean['Position_Group'] == 'Defender']):,})", fontsize=12.5, weight='bold', pad=12)

    # (c) Nhóm Midfielder
    corr_mid = df_clean[df_clean['Position_Group'] == 'Midfielder'][features].corr()
    sns.heatmap(
        corr_mid, annot=True, fmt='.2f', cmap='coolwarm',
        vmin=-1.0, vmax=1.0, ax=axes[1, 0],
        linewidths=0.8, linecolor='white',
        annot_kws={'size': 10, 'weight': 'bold'},
        cbar_kws={'label': 'Hệ số r'}
    )
    axes[1, 0].set_title(f"(c) Nhóm Tiền vệ - Midfielder (n = {len(df_clean[df_clean['Position_Group'] == 'Midfielder']):,})", fontsize=12.5, weight='bold', pad=12)

    # (d) Nhóm Forward
    corr_fwd = df_clean[df_clean['Position_Group'] == 'Forward'][features].corr()
    sns.heatmap(
        corr_fwd, annot=True, fmt='.2f', cmap='coolwarm',
        vmin=-1.0, vmax=1.0, ax=axes[1, 1],
        linewidths=0.8, linecolor='white',
        annot_kws={'size': 10, 'weight': 'bold'},
        cbar_kws={'label': 'Hệ số r'}
    )
    axes[1, 1].set_title(f"(d) Nhóm Tiền đạo - Forward (n = {len(df_clean[df_clean['Position_Group'] == 'Forward']):,})", fontsize=12.5, weight='bold', pad=12)

    fig.suptitle("HÌNH 4.1: MA TRẬN HỆ SỐ TƯƠNG QUAN PEARSON TOÀN BỘ VÀ THEO TỪNG NHÓM VỊ TRÍ", fontsize=14.5, weight='bold', color='#1f2937', y=0.995)
    plt.tight_layout()

    heatmap_file = OUTPUT_DIR / "hinh_4_1_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"=> Đã lưu thành công Biểu đồ 1: {heatmap_file} (DPI 300)")

    # ------------------------------------------------------------------------------
    # Hàm hỗ trợ vẽ elip tin cậy đa biến 95% (Confidence Ellipse 95%)
    # ------------------------------------------------------------------------------
    def plot_confidence_ellipse(x, y, ax, n_std=2.4477, facecolor='none', edgecolor='black', **kwargs):
        """
        Vẽ elip tin cậy 95% cho 2 biến ngẫu nhiên (n_std = sqrt(chi2.ppf(0.95, df=2)) = 2.4477)
        """
        if len(x) != len(y) or len(x) < 3:
            return
        
        cov = np.cov(x, y)
        denom = np.sqrt(cov[0, 0] * cov[1, 1])
        pearson = cov[0, 1] / denom if denom > 0 else 0.0
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor=facecolor, edgecolor=edgecolor, **kwargs)
        
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)
        
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    # ------------------------------------------------------------------------------
    # Biểu đồ 2: Scatter plot phân tách Defending vs Shooting + Elip tin cậy 95%
    # ------------------------------------------------------------------------------
    fig, ax2 = plt.subplots(figsize=(10, 8), dpi=300)

    for g in groups:
        sub = df_clean[df_clean['Position_Group'] == g]
        ax2.scatter(
            sub['Defending'], sub['Shooting'],
            color=palette_colors[g],
            label=f"{g} (n={len(sub):,})",
            alpha=0.35, s=16, edgecolors='none'
        )

    for g in groups:
        sub = df_clean[df_clean['Position_Group'] == g]
        x_vals = sub['Defending'].values
        y_vals = sub['Shooting'].values
        mean_x = np.mean(x_vals)
        mean_y = np.mean(y_vals)
        
        plot_confidence_ellipse(
            x_vals, y_vals, ax=ax2,
            n_std=2.4477, # chi2 95% df=2
            edgecolor=palette_colors[g],
            facecolor=palette_colors[g],
            alpha=0.15, linewidth=2.5, linestyle='--'
        )
        
        ax2.scatter(
            mean_x, mean_y,
            color=palette_colors[g], s=180,
            marker='X', edgecolors='black', linewidths=1.8, zorder=10
        )
        
        ax2.annotate(
            f"Tâm {g}\n({mean_x:.1f}, {mean_y:.1f})",
            (mean_x, mean_y),
            textcoords="offset points", xytext=(0, 14),
            ha='center', fontsize=10.5, weight='bold', color='#111827',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=palette_colors[g], lw=1.5, alpha=0.9),
            zorder=11
        )

    ax2.set_title("HÌNH 4.2: PHÂN TÁCH KHÔNG GIAN DEFENDING - SHOOTING VÀ ELIP TIN CẬY 95%", fontsize=13.5, pad=16, weight='bold', color='#1f2937')
    ax2.set_xlabel("Chỉ số Phòng ngự (Defending)", fontsize=12, labelpad=10, weight='bold')
    ax2.set_ylabel("Chỉ số Dứt điểm (Shooting)", fontsize=12, labelpad=10, weight='bold')
    ax2.set_xlim(5, 100)
    ax2.set_ylim(5, 100)
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax2.legend(
        title="Nhóm vị trí thi đấu", title_fontsize='11', fontsize='10',
        loc='upper right', frameon=True, framealpha=0.95,
        facecolor='white', edgecolor='#cbd5e1'
    )

    plt.tight_layout()
    scatter_file = OUTPUT_DIR / "hinh_4_2_scatter_ellipse.png"
    plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"=> Đã lưu thành công Biểu đồ 2: {scatter_file} (DPI 300)")

    # ------------------------------------------------------------------------------
    # A1: Biểu đồ 3 (Pairplot 6 biến, n = 1,500, random_state=42)
    # ------------------------------------------------------------------------------
    print("\n--- Đang tạo Hình 4.3: Pairplot 6 biến (mẫu ngẫu nhiên n = 1,500)... ---")
    df_sample = df_clean.sample(n=min(1500, len(df_clean)), random_state=42)

    g_pair = sns.pairplot(
        df_sample,
        vars=features,
        hue='Position_Group',
        hue_order=groups,
        palette=palette_colors,
        diag_kind='kde',
        plot_kws={'alpha': 0.5, 's': 18, 'edgecolor': 'none'},
        diag_kws={'alpha': 0.4, 'linewidth': 1.5}
    )
    g_pair.fig.suptitle("HÌNH 4.3: MA TRẬN PHÂN BỐ TẬP HỢP (PAIRPLOT) GIỮA 6 BIẾN KỸ THUẬT (MẪU N = 1,500)", y=1.02, fontsize=14, weight='bold', color='#1f2937')

    pairplot_file = OUTPUT_DIR / "hinh_4_3_pairplot.png"
    g_pair.savefig(pairplot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"=> Đã lưu thành công Biểu đồ 3: {pairplot_file} (DPI 300)")

    # ------------------------------------------------------------------------------
    # A3: Biểu đồ 4 (Q-Q plot Mahalanobis bình phương theo từng nhóm so với Chi2 df=6)
    # ------------------------------------------------------------------------------
    print("\n--- Đang tạo Hình 4.4: Q-Q plot khoảng cách Mahalanobis bình phương theo từng nhóm... ---")
    fig_qq, axes_qq = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    for idx, g in enumerate(groups):
        sub_vals = df_clean[df_clean['Position_Group'] == g][features].values
        n_g = len(sub_vals)
        mean_g = np.mean(sub_vals, axis=0)
        cov_g = np.cov(sub_vals, rowvar=False)
        diff_g = sub_vals - mean_g
        
        # B2: Dùng np.linalg.solve thay cho inv(S)
        try:
            y_g = np.linalg.solve(cov_g, diff_g.T).T
            d2_g = np.sum(diff_g * y_g, axis=1)
        except np.linalg.LinAlgError:
            cov_g_pinv = np.linalg.pinv(cov_g)
            d2_g = np.sum((diff_g @ cov_g_pinv) * diff_g, axis=1)
            
        d2_sorted = np.sort(d2_g)
        
        # Tính phân vị lý thuyết Chi-bình phương (bậc tự do p = 6)
        quantiles_prob = (np.arange(1, n_g + 1) - 0.5) / n_g
        theoretical_q = chi2.ppf(quantiles_prob, df=p)
        
        ax = axes_qq[idx]
        ax.scatter(
            theoretical_q, d2_sorted,
            color=palette_colors[g], alpha=0.45, s=12,
            label=f"{g} (n={n_g:,})"
        )
        
        # Đường tham chiếu chuẩn lý thuyết 45 độ (y = x)
        max_limit = min(float(theoretical_q.max()), float(d2_sorted.max())) * 1.1
        ax.plot([0, max_limit], [0, max_limit], color='#dc2626', linestyle='--', linewidth=2, label='Đường chuẩn Chi2 (y = x)')
        
        ax.set_title(f"Q-Q Plot Chi2 (df=6): Nhóm {g}", fontsize=12, weight='bold', color='#1f2937')
        ax.set_xlabel("Phân vị lý thuyết Chi-bình phương (df=6)", fontsize=10.5, weight='bold')
        ax.set_ylabel("Khoảng cách Mahalanobis bình phương ($D_M^2$)", fontsize=10.5, weight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=True, facecolor='white', loc='upper left')

    fig_qq.suptitle("HÌNH 4.4: Q-Q PLOT KIỂM TRA PHÂN PHỐI CHUẨN ĐA BIẾN THEO TỪNG NHÓM VỊ TRÍ", fontsize=14, weight='bold', color='#1f2937', y=1.03)
    plt.tight_layout()

    qq_file = OUTPUT_DIR / "hinh_4_4_qq_mahalanobis.png"
    plt.savefig(qq_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"=> Đã lưu thành công Biểu đồ 4: {qq_file} (DPI 300)")

    print("\n" + "=" * 80)
    print("HOÀN THÀNH TOÀN BỘ CHƯƠNG TRÌNH PHÂN TÍCH THỐNG KÊ ĐA BIẾN!")
    print(f"Toàn bộ bảng số liệu đã được lưu trong: {RESULTS_DIR.resolve()}")
    print(f"Toàn bộ file hình ảnh đã được lưu trong: {OUTPUT_DIR.resolve()}")
    print("=" * 80)

finally:
    # BƯỚC 5: Luôn đảm bảo flush, close logger và khôi phục stdout
    logger.flush()
    logger.close()
    sys.stdout = logger.terminal
