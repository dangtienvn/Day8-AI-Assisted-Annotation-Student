# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Đặng Thanh Tiến - MSSV: 2A202602099

Công cụ gán nhãn đã dùng: CVAT Docker Local (bản v2.76.0)

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian (temporal split) và chèn vùng đệm ở giữa thay vì chia ngẫu nhiên (random split) để tránh hiện tượng **rò rỉ dữ liệu qua chuỗi thời gian (temporal data leakage)**.

Trong video đường cao tốc ban đêm, các khung hình liên tiếp có tính tương quan chuỗi rất cao về bối cảnh đường phố, góc quay camera, điều kiện chiếu sáng và vị trí các xe đang di chuyển. Nếu chia ngẫu nhiên, các ảnh trong tập train và test sẽ gần như trùng lặp nội dung. Khi đó, số đo trên tập kiểm thử sẽ bị **lệch theo hướng quá lạc quan (overly optimistic / inflated metrics)**, khiến chúng ta tưởng mô hình hoạt động rất tốt nhưng thực tế khi triển khai sang thời điểm khác thì mô hình sẽ thất bại. Việc chia theo trục thời gian và giữ vùng đệm giúp đánh giá chính xác khả năng tổng quát hóa của mô hình trên dữ liệu tương lai.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 từ `reports/rounds_table.md`:
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào `outputs/compare_round0.jpg`, mô hình khởi đầu lạnh (YOLOv8n huấn luyện trên tập COCO) gặp khó khăn và không khớp nhãn tham chiếu ở các dạng xe:

1. **Xe ô tô ở xa có kích thước nhỏ**: Các xe ở góc xa hoặc mới nhô ra khỏi khúc cua.
2. **Xe bị chói đèn pha ban đêm**: Ánh đèn pha mạnh khiến viền thân xe bị nhòa vào nền tối.
3. **Xe tải / Đầu kéo lớn**: Kích thước thân xe khác biệt với ô tô con chuẩn.

Độ phủ (Recall) theo kích thước xe thể hiện rõ nhược điểm này:

- `R small` chỉ đạt **0.182 (18.2%)** trên 66 box tham chiếu, chứng tỏ mô hình bỏ sót gần 82% các xe cỡ nhỏ ở xa.
- `R medium` đạt **0.547 (54.7%)** và `R large` đạt **0.561 (56.1%)**.

Trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai: Khi gặp các xe ở khoảng cách rất xa có kích thước dưới hoặc xấp xỉ 16 pixel (ngưỡng `ref_ignored_tiny`). Nhãn tham chiếu do mô hình tự động tạo ra có thể đã bỏ sót các xe này hoặc nhận nhầm vệt đèn đường; người rà soát cần soi lại ảnh gốc bằng mắt để kiểm tra tính đúng đắn của nhãn tham chiếu.

## 3. Chiến lược chọn mẫu

Công thức tính điểm chọn mẫu: $score = W_U \cdot U + W_A \cdot A + W_D \cdot D$

- $U$ (Uncertainty - Độ bất định): Đo mức độ phân vân của mô hình trên các box dự đoán (dựa trên confidence entropy/margin). Ảnh có $U$ cao tức là mô hình chưa tự tin.
- $A$ (Ambiguity / Density - Mức độ phức tạp/mật độ): Phản ánh số lượng đối tượng nhập nhằng và mật độ giao thông trong khung hình.
- $D$ (Diversity / Spatial distance - Độ đa dạng không gian): Đo khoảng cách phân bố vị trí không gian của các box để tránh chọn các ảnh có bố cục giống hệt nhau.
- `MIN_GAP_S` (Ngưỡng khoảng cách thời gian tối thiểu): Đóng vai trò nén thời gian (temporal suppression). Lệnh này chặn không cho chọn các ảnh nằm quá gần nhau về mặt thời gian (ví dụ trong khoảng 2 giây), nhằm tránh lãng phí ngân sách rà nhãn vào các ảnh gần như trùng lặp nội dung.

Dẫn chứng từ `reports/SELECTION.md`:

- **3 frame được mô hình chọn trong lô 12 ảnh**: `frame_0182.jpg` (Rank 1, Score=0.9591, U=0.9182), `frame_0380.jpg` (Rank 3, Score=0.9170, U=0.9340), `frame_0326.jpg` (Rank 4, Score=0.9155, U=0.9310).
- **1 frame có điểm cao nhưng không chọn**: `frame_0372.jpg` (Rank 6, Score=0.9101, t=148.8s) bị loại bỏ vì nằm quá gần `frame_0369.jpg` (Rank 2, t=147.6s), kích hoạt bộ lọc `MIN_GAP_S`.

Điểm bất định $U$ **không chứng minh** rằng ảnh đó chắc chắn sẽ làm cải thiện mô hình. Điểm $U$ cao chỉ cho biết mô hình đang thiếu tự tin tại khung hình đó. Nếu ảnh chứa quá nhiều nhiễu (như ánh đèn phản chiếu phức tạp) hoặc nhãn do người rà soát gán bị sai sót, việc fine-tune trên ảnh đó thậm chí có thể làm giảm hiệu năng chung của mô hình.

## 4. Các vòng học chủ động (active learning)

Bảng tổng hợp từ `reports/rounds_table.md`:
| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tuned (round1 12 images) | 12 | 350 | 0.785 | +0.014 | 0.936 | 0.509 | 0.659 | 0.212 | 0.564 | 0.585 |

Trình bày chi tiết Vòng 1:

- **Mức độ sửa nhãn gợi ý (`outputs/round1_diff.md`)**: Lô 12 ảnh có 169 box do model đề xuất. Sau khi rà soát và bổ sung nhãn trên CVAT local thu được 350 box final (Accepted: 161, Edited: 2, Deleted: 6 box giả do vệt phản chiếu, Added: 187 box xe bị sót).
- **Thay đổi AP50**: AP50 tăng **+0.014** (từ 0.771 lên 0.785).
- **Nhóm xe cải thiện**: Tăng trưởng rõ nhất ở nhóm xe cỡ nhỏ `R small` tăng từ **0.182 lên 0.212 (+3.0%)** và `R large` tăng từ **0.561 lên 0.585 (+2.4%)**.

Dựa trên `outputs/compare_round1.jpg`, tại `frame_0099.jpg`, mô hình sau fine-tune đã phát hiện được các xe nhỏ phía xa góc trên khúc cua (nơi Vòng 0 bỏ sót). Quyết định này hoàn toàn khớp với quan sát độc lập ban đầu trong [`reports/BLIND_SCAN.md`](BLIND_SCAN.md) và ca bổ sung nhãn trong [`reports/REVIEW_LOG.csv`](REVIEW_LOG.csv).

Mô tả ca khó theo guideline (`frame_0331.jpg`): Xe chạy ban đêm chiếu chói đèn pha xuống mặt đường. Theo đúng [GUIDELINE_LABEL.md](../GUIDELINE_LABEL.md), box được điều chỉnh chỉ ôm sát phần viền thân xe nhìn thấy, không được vẽ trùm lên vệt sáng chiếu xuống mặt đường phía trước.

## 5. Kết luận và giới hạn

**Đánh giá kết quả**: Việc gán nhãn bổ sung 12 ảnh chọn lọc qua Active Learning đã giúp cải thiện AP50 (+0.014) và nâng cao độ phủ cho các xe ở xa (Recall small +3.0%).

**Quyết định**: Dừng lại ở Vòng 1 để nộp bài vì mức tăng AP50 đã đạt yêu cầu thực hành và ngân sách thời gian rà nhãn tối ưu. Nếu làm tiếp Vòng 2, đề xuất ưu tiên các ảnh có độ bất định cao ở các khoảng thời gian chưa được khai thác (t=0s - 30s) như `frame_0002.jpg` (Rank 18).

**Tác động của các giới hạn**:

1. **Tập kiểm thử nhỏ (20 ảnh)**: Dễ gây ra biến động ngẫu nhiên trong chỉ số AP50 (chỉ cần phát hiện thêm 1-2 box là số đo thay đổi đáng kể).
2. **Luật bỏ qua xe quá nhỏ (<16px)**: Giúp giảm nhiễu đánh giá nhưng cũng bỏ qua các ca biên ở cực xa.
3. **Nhãn tham chiếu do mô hình tạo**: Chưa được con người rà soát 100%, nên chỉ số AP50 thực chất là đo _độ trùng khớp với tập tham chiếu AI_ chứ không phải là độ chính xác tuyệt đối so với thực địa.

**Quy trình kiểm tra nếu AP50 giảm**: Nếu AP50 giảm ở các vòng sau, tôi sẽ kiểm tra lại `outputs/round1_diff.md` và `reports/REVIEW_LOG.csv` để phát hiện xem có sự bất nhất trong quy chuẩn gán nhãn (inconsistent labeling) giữa các ảnh hay không, kiểm tra xem có vẽ trùm vệt sáng đèn hoặc gộp nhầm box xe hay không trước khi tiếp tục huấn luyện.
