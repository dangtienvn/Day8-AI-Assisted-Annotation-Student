# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, 5 frame được ưu tiên nếu chỉ có ngân sách rà 5 ảnh bao gồm:
1. `frame_0182.jpg` (Rank 1, t=72.8s, Score=0.9591, U=0.9182, A=1.0, D=1.0): Mức bất định cao, chứa 18 box bất định (n_ambiguous=18) trong tổng số 28 box, đại diện cho đoạn giao thông đông và chói đèn.
2. `frame_0369.jpg` (Rank 2, t=147.6s, Score=0.9324, U=0.9315, A=0.8889, D=1.0): Mật độ xe cao (43 box), U cực lớn (0.9315), đại diện cho khoảng thời gian t~147s.
3. `frame_0331.jpg` (Rank 5, t=132.4s, Score=0.9154, U=0.8308, A=1.0, D=1.0): Mật độ 47 box cao nhất lô, độ phủ không gian D=1.0 và A=1.0, nhiều xe bị vệt đèn chiếu ngang.
4. `frame_0099.jpg` (Rank 8, t=39.6s, Score=0.9063, U=0.9460, A=0.7778, D=1.0): U rất cao (0.9460) ở khoảng thời gian sớm (t=39.6s), giúp mở rộng độ đa dạng mốc thời gian; quan sát mắt cho thấy có xe mới nhô ra ở khúc cua xa bị bỏ sót.
5. `frame_0392.jpg` (Rank 15, t=156.8s, Score=0.8874, U=0.9747, A=0.6667, D=1.0): Sở hữu chỉ số bất định U cao nhất tập (0.9747), chốt lại giai đoạn cuối video.

Ba frame thuộc lô 12 ảnh model chọn:
- `frame_0182.jpg` (Rank 1): Score=0.9591, U=0.9182, A=1.0, D=1.0, 28 box. Bằng chứng CSV chứng minh độ bất định và tính đa dạng không gian tối đa.
- `frame_0380.jpg` (Rank 3): Score=0.9170, U=0.9340, 40 box. Bằng chứng contact sheet cho thấy vệt phản chiếu đèn đường trên dải phân cách khiến AI phân vân.
- `frame_0326.jpg` (Rank 4): Score=0.9155, U=0.9310, 39 box, 15 box bất định.

Một frame có điểm cao nhưng không chọn:
- `frame_0372.jpg` (Rank 6, Score=0.9101, t=148.8s): Dù có điểm tổng hợp đứng thứ 6 nhưng bị hệ thống đặt `selected=False` do cơ chế nén khoảng cách thời gian (`MIN_GAP_S`), vì nằm quá gần `frame_0369.jpg` (t=147.6s, Rank 2). Việc bỏ qua ảnh này giúp tránh lãng phí công rà soát vào các ảnh trùng lặp bối cảnh.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:
- Điểm bất định cao ($U$) chỉ phản ánh việc mô hình chưa tự tin tại khung hình đó, chứ không bảo đảm 100% rằng việc fine-tune trên ảnh đó sẽ làm tăng mAP/AP50 trên toàn bộ tập test. Việc chọn mẫu active learning cũng chưa thể loại trừ lỗi do con người gán nhãn sai hoặc tập test quá nhỏ (20 ảnh).
