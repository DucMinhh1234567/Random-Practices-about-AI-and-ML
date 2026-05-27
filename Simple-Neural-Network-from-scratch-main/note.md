## Use case:
- Sử dụng dataset MNIST để train mô hình neural network để phân loại chữ số từ 0 đến 9
- với mỗi ảnh là 28x28 pixel, tương ứng với 784 features
- Các giá trị chạy từ 0 đến 255 với 0 là đen 225 là trắng
- Represent là một ma trận với size là m x 784, sau đó dùng chuyển vị (A^T) để chuyển thành ma trận 784 x m (để làm gì??? -> à chắc để thành cột làm vector input dễ hơn)

## Neural Network:
Cấu trúc cơ bản thường có 3 lớp: input layer, hidden layer, output layer.
- Input layer (784 nodes): lớp đầu vào nhận feature (pixel) của ảnh, mỗi feature tương ứng 1 node
- Hidden layer (10 nodes): lớp ẩn nhận output từ input layer, và output ra 10 nodes tương ứng với 10 chữ số
- Output layer (10 nodes): lớp đầu ra nhận output từ hidden layer, và output ra 10 nodes tương ứng với 10 chữ số

## Cách hoạt động:
#### Bước 1: Foward Propagation:
- Về cơ bản là lớp trước chạy ra output rồi truyền cho lớp sau

Ví dụ, lớp đầu vào sẽ cung cấp dữ liệu cho lớp ẩn. Lớp ẩn sẽ xử lý dữ liệu và xuất kết quả về lớp đầu ra. Lớp đầu ra sẽ xuất ra kết quả cuối cùng.

A0 = X (784 x m) - (input layer)
Z1 = W1 * A0 + b1 (hidden layer) - unactivated output
A1 = ReLu(Z1) (hidden layer) - activated output

- Activation function: là hàm số để kích hoạt các node, thường là hàm sigmoid, ReLU, tanh, softmax, ...
lý do cần: để đảm bảo output của node luôn nằm trong khoảng 0 và 1, hoặc -1 và 1, hoặc 0 và 1, ... hoặc đơn giản là để làm cho nó phi tuyến tính, bởi vì nếu không thì các lớp sau chỉ là sự kết hợp tuyến tính của các nút trước đó không đủ khả năng biểu đạt để dự đoán bất cứ điều gì. Đại khái là nếu không có các hàm này thì mỗi node chỉ là một kết quả tuyến tính của các node trước đó, không có ý nghĩa gì

- trong layer 2 này sẽ dùng ReLu:
- ReLu(x) = x if x > 0 else 0

Z2 = W2 * A1 + b2 (output layer) - unactivated output
A2 = softmax(Z2) (output layer) - activated output

#### Bước 2: BackPropagation:
Sau khi forward propagation xong, sẽ cho ra kết quả. Nhưng khả năng rất cao là kết quả đó không chính xác, vì vậy cần phải cập nhật lại các trọng số W và bias b để kết quả đó chính xác hơn. -> Do vậy dùng Backward Propagation để cập nhật lại các trọng số W và bias b.

Cách hoạt động:
- Sau khi NN dự đoán xong, sẽ so sánh kết quả với giá trị thực tế để tính toán sai số (loss)
- Sau đó tìm hiểu xem các node, node nào có đóng góp vào việc làm cho dự đoán bị sai sau đó ta sửa các tham số đó

dZ2 (10 x m) = A2 - Y (10 x m)
với Y là ma trận one-hot encoding của giá trị thực tế, ví dụ:
- Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] -> chữ số 9

Sau đó tính toán xem W và b đóng góp bao nhiêu cho lỗi đó bằng cách tính gradient của loss với W và b. (hay tính đạo hàm riêng của loss với W và b)

dW2 (10 x 10) = (1 / m) * dZ2 * A1^T (10 x m)
db2 (10 x 1) = (1 / m) * sum(dZ2)

Các bước trên sẽ tính được layer output đã dự đoán lệch bao nhiêu so với giá trị thực tế, sau đó sửa các trọng số W và bias b để làm giảm lỗi đó.

Tương tự với hidden layer:

dZ1 (10 x m) = W2^T * dZ2 * deriv_relu(Z1)

Lưu ý: do khi ta lan truyền tiến, đã dùng relu nên cũng cần tính đạo hàm của hàm ReLU để tính dZ1

dW1 (10 x 784) = (1 / m) * dZ1 * X^T (784 x m)
db1 (10 x 1) = (1 / m) * sum(dZ1)

#### Bước 3: Cập nhật tham số:
Sau khi tính toán lan truyền ngược xong, ta sẽ xác định được các trọng số W và bias b của từng layer đóng góp bao nhiêu cho lỗi đó, sau đó cập nhật lại các trọng số W và bias b đó.

Khi cập nhật trọng số, ta sẽ có một biến tên `alpha` (learning rate) để quyết định tốc độ cập nhật trọng số. Giá trị của biến này càng cao thì sẽ thay đổi trọng số W và bias b càng nhiều và ngược lại, nhưng nếu quá cao thì có thể sẽ không hội tụ. Hiểu đơn giản học nhanh quá thì ko đọng lại kiến thức mà học chậm quá thì ko học được gì.

W1 = W1 - alpha * dW1
b1 = b1 - alpha * db1
W2 = W2 - alpha * dW2
b2 = b2 - alpha * db2

Ngoài ra còn có biến `iter` (interations) để quyết định số lần mô hình học. Tương tự như trên, nếu `iter` càng cao thì mô hình học càng nhiều và ngược lại, nhưng nếu quá cao thì mô hình sẽ có khả năng bị overfitting hoặc quá thấp thì underfitting. Đại khái là học quá nhiều thì lú còn học ít quá thì chưa kịp tiếp thu kiến thức.

#### Bước 4: Lặp lại:

## Lưu ý:
Mô hình trên là mô hình cơ bản nhất của Neural Network, vẫn còn nhiều phần có thể cải thiện hoặc các kỹ thuật có thể áp dụng, ví dụ như:

- Thêm optimizer mạnh mẽ hơn để tăng tốc độ học của mô hình
- Thêm acceleration technique để giảm dần learning rate của mô hình học để học nhanh khi bắt đầu và tinh chỉnh về sau
- Chưa có tính loss hoặc vai trò của loss chưa quá rõ ràng


## Bước tiếp theo: NLP
Cấp độ 1 (Cơ bản): Sử dụng mạng Dense (ANN) thông thường kết hợp với kỹ thuật Bag-of-Words hoặc TF-IDF để làm quen với dữ liệu văn bản.
Cấp độ 2 (Nâng cao hơn): Sử dụng mạng RNN (Recurrent Neural Network) hoặc LSTM. Đây là những kiến trúc "sinh ra" dành cho NLP vì chúng có khả năng ghi nhớ thứ tự của các từ trong câu.