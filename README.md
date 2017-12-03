1. Tiền xử lí văn bản 
- chuyển về chữ thường, loại bỏ dấu câu, loại bỏ kí tự đặc biệt, stopword...
- Giảm số chiều vecto sử dụng PCA

Chú ý:
- File p4.sqlite có 23000 row, khi lấy random 10000 dòng ngẫu nhiên thì số vocabulary khoảng 94893.

2. Đánh giá
- TF-IDF
               precision    recall  f1-score   support

     business       0.87      0.82      0.84      4532
entertainment       0.69      0.92      0.79      2458
       health       0.55      0.83      0.66       369
       sports       0.98      0.97      0.98     10969
     politics       0.81      0.51      0.62      1672

  avg / total       0.90      0.89      0.89     20000



- BOW
               precision    recall  f1-score   support

     business       0.87      0.78      0.82      4539
entertainment       0.70      0.91      0.79      2463
       health       0.47      0.83      0.60       330
       sports       0.97      0.97      0.97     11035
     politics       0.81      0.55      0.65      1633

  avg / total       0.89      0.88      0.88     20000
  
  
3. Độ đo
Precision: trong tập tìm được thì bao nhiêu cái (phân loại) đúng.
Recall: trong số các tồn tại, tìm ra được bao nhiêu cái (phân loại).