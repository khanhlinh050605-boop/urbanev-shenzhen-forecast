# ev_pipeline

Tách từ `ev_pipeline_v4.py`. Chạy từ thư mục chứa package:

    python -m ev_pipeline --data ev_pipeline/raw_data --out output [--seed 42] [--reuse-saved-predictions]

Dữ liệu không nằm trong repo. Trước khi chạy, tải bộ dữ liệu UrbanEV (Thâm Quyến) và chép các file CSV vào thư mục `ev_pipeline/raw_data/` (hoặc đổi đường dẫn sau `--data`):

- GitHub: https://github.com/IntelligentSystemsLab/UrbanEV
- Bản lưu trữ Dryad: https://datadryad.org/dataset/doi:10.5061/dryad.np5hqc04z

Yêu cầu Python 3.12 (đã kiểm tra với 3.12.10); cài thư viện bằng `pip install -r requirements_py312.txt`. `--reuse-saved-predictions` so sánh `fit_signature` (mã băm AST), nên kết quả phụ thuộc phiên bản Python: trên Python 3.13 trở lên chữ ký có thể khác và lệnh báo `Saved prediction configuration or input mismatch`.

| Package | Vai trò |
|---|---|
| `config.py` | `Settings`, hằng số giao thức (lịch, folds, tham số mô hình khoá) |
| `cli.py`, `pipeline.py` | Điểm vào CLI; các stage: setup → clean_data → models → evaluate → decision_analysis → visualize → finish |
| `utils/` | Ghi JSON / npy / csv / progress |
| `data/` | Đọc CSV, kiểm tra toàn vẹn, provenance, thống kê mô tả |
| `features/` | Lịch nghỉ lễ, xây feature, chuẩn hoá theo tập train, ma trận mô hình |
| `scoring/` | Chia fold, `Evaluator` (metric zone/city), bootstrap |
| `models/` | Baseline, LightGBM/XGBoost, ridge, vòng fit chính, cluster, extended fits |
| `analysis/` | Đánh giá dự báo, calibration, phân tích quyết định, leakage, so sánh bảng tham chiếu |
| `benchmarks/` | Mã upstream (dạng chuỗi), chạy subprocess thống kê, benchmark chuẩn hoá |
| `reporting/` | Manifest, `fit_signature`, hash predictions |
| `viz/` | `style`, `context` (trạng thái + helper dùng chung), `figures/` (mỗi hình một file), `runner` |

Lưu ý: `--seed` phải được áp dụng trước khi import các module khác (các default như `seed=SEED`
gắn lúc import), nên `cli.main` gọi `config.set_seed` rồi mới import `pipeline`.

Hình: `viz/merged.py` vẽ các hình ghép (kích thước khổ rộng 183 mm) từ `output/results/*.csv`. Các hình 3, 4, 7, 8, 9, 10 của report được vẽ lại ở đúng độ rộng in 160 mm bằng `viz/report_figures.py`, nên chèn vào LaTeX với `width=\textwidth` là kích thước 1:1:

    python -m ev_pipeline.viz.report_figures output/results output/figures

Lệnh này ghi `fig3`, `fig4`, `fig7`, `fig8`, `fig9`, `fig10` (`.png` và `.pdf`) vào `output/figures/`; các hình này cần các file `figure_*.csv`, `zone_failure_*.csv`, `ablation_steps.csv` và `cluster_profiles.csv` trong `output/results/`.
