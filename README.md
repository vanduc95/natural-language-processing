1. Tiền xử lí văn bản 
- chuyển về chữ thường, loại bỏ dấu câu, loại bỏ kí tự đặc biệt, stopword...
- Giảm số chiều vecto sử dụng PCA

Chú ý:
- File p4.sqlite có 23000 row, khi lấy random 10000 dòng ngẫu nhiên thì số vocabulary khoảng 94893.

2. Đánh giá

5 class: business, entertainment, politics, sports, technology.
+ training: p1.sqlite.  3000 bản ghi đầu cho mỗi class
+ test: p1.sqlite.    2000 bản ghi tiếp theo cho mỗi class

2.1. k-NN / BOW
                precision    recall  f1-score   support
     business       0.90      0.14      0.25      2000
entertainment       0.32      0.82      0.46      2000
   technology       0.88      0.23      0.36      2000
       sports       0.46      0.67      0.55      2000
     politics       0.95      0.54      0.69      2000

  avg / total       0.70      0.48      0.46     10000
  
  
2.2 k-NN / TF-IDF
               precision    recall  f1-score   support
     business       0.92      0.67      0.77      2000
entertainment       0.88      0.86      0.87      2000
   technology       0.86      0.84      0.85      2000
       sports       0.85      0.96      0.90      2000
     politics       0.81      0.97      0.88      2000

  avg / total       0.86      0.86      0.85     10000


2.3. SVM / BOW
                precision    recall  f1-score   support
     business       0.92      0.74      0.82      2000
entertainment       0.89      0.94      0.91      2000
   technology       0.85      0.93      0.89      2000
       sports       0.91      0.97      0.94      2000
     politics       0.93      0.92      0.93      2000

  avg / total       0.90      0.90      0.90     10000



2.4. SVM / TF-IDF
                precision    recall  f1-score   support
     business       0.94      0.74      0.83      2000
entertainment       0.91      0.95      0.93      2000
   technology       0.85      0.95      0.89      2000
       sports       0.94      0.99      0.96      2000
     politics       0.94      0.94      0.94      2000

  avg / total       0.92      0.91      0.91     10000



3. Độ đo
Precision: trong tập tìm được thì bao nhiêu cái (phân loại) đúng.
Recall: trong số các tồn tại, tìm ra được bao nhiêu cái (phân loại).

Sử dụng hold-out để đánh giá

4. Thử nghiệm
4.1 Với 1000 bản ghi cho mỗi class
- Thời gian cho KNN là 42min


4.2 Với 2000 bản ghi cho mỗi class
- Thời gian cho KNN là 220min


4.2 Với 3000 bản ghi cho mỗi class
- Thời gian cho KNN là 575min, SVM 85min

KNN chay lâu hơn SVM


5. Trang web để test: http://abcnews.go.com/


























