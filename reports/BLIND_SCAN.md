# Quét độc lập trước khi xem pre-label

Frame: frame_0099.jpg

Số xe nhìn thấy bằng mắt: 20

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:
1. Nhóm 3 xe ô tô ở phía xa phía trên góc trái (hướng đi từ trên xuống), kích thước nhỏ và mới nhô ra một chút ở khu vực khúc cua, dễ bị AI bỏ sót do độ phân giải thấp.
2. Cụm các xe ở phía xa khu vực trung tâm có ánh đèn pha chói, khoảng cách giữa các xe hẹp, dễ bị AI vẽ gộp nhiều xe vào cùng một bounding box hoặc vẽ lệch kích thước.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
