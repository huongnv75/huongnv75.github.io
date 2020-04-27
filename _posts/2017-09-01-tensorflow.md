---
title: Giới thiệu về Tensorflow
author: huongnv75
date: 2017-09-01 12:00:00 +0700
categories: [Blogging, Tensorflow Tutorial]
tags: [tensorflow]
---

Xin chào các bạn. Những chuỗi ngày tiếp theo, mình sẽ giới thiệu các bạn về một open source đang khá `hot` hiện nay. Đó chính là ông Tensorflow này. Nói qua một chút thì Tensorflow là open source được sinh ra từ ông chủ lớn google từ cuối năm 2015, nó là một framework, cung cấp các thư viện phục vụ cho việc tính toán, hình thành các mô hình cũng như việc training cho các bài toán về machine learning, deep learning. Trước khi đi vào bài, chúng ta bắt đầu làm quen với những khái niệm hơi khó hiểu của Tensorflow, mình dùng văn viết để các bạn dễ hiểu nhé.

## 1. Các khái niệm cơ bản
### 1.1. Node
Tại sao lại là node? Node là gì? Để trả lời được các câu hỏi này thì bạn cần biết rằng là Tensorflow hoạt động theo kiểu dòng chảy của dữ liệu. Do đó, node chính là điểm giao cắt trong graph đó. Tại sao điều này quan trọng, vì node chính là đại diện cho việc thay đổi của dữ liệu, nên việc lưu trữ lại tham chiếu của các Node này là rất quan trọng.
### 1.2. Tensor
Vậy sau khi có graph như trên, tức là có dòng chảy, vậy thì dòng chảy đó chảy cái gì? Trong Tensorflow, mọi kiểu dữ liệu đều được quy về một đối tượng, đối tượng đó được gọi là Tensor. Tensor là một kiểu dữ liệu dạng mảng có nhiều chiều, đồng thời mảng nhiều chiều này được đính kèm thêm một vài thuộc tính tham số cần thiết. Các thuộc tính đó được mô tả như sau:

* **device**: Tên của thiết bị mà Tensor hiện tại sẽ được xuất bản. Có thể là None.
* **graph**: Đồ thị chứa Tensor hiện tại.
* **name**: Tên của Tensor hiện tại.
* **shape**: TensorShape mô tả lại Shape của Tensor hiện tại.
* **op**: Toán tử được sử dụng để xuất bản Tensor hiện tại.
* **dtype**: Kiểu của các phần tử trong Tensor hiện tại.

### 1.3. Rank
Rank là bậc hay độ sâu của một Tensor. Ví dụ như `Tensor = [2017]` sẽ có rank là 1, `Tensor = [[[1,1,1],[0,2,4]]]` sẽ có rank bằng 3, `Tensor = [[2,1,1],[8,-3,4]]` sẽ có rank bằng 2.  Việc phân rank này khá quan trọng vì nó đồng thời cũng giúp phân loại dữ liệu của Tensor. Khi ở cách rank đặc biệt cụ thể, Tensor có những tên riêng như sau:

* **Scalar**: Khi Tensor có rank bằng 0, Tensor đại diện cho một số hoặc một chuỗi cụ thể. Ví dụ: `scalar=2017`.
* **Vector**: Vector là một Tensor có rank bằng 1. Trong python thì Vector là một list hay mảng một chiều chứa các số. Ví dụ: `list=[123,234,345]`.
* **Matrix**: Đây là một Tensor có rank bằng 2 hay mảng hai chiều theo khái niệm của Python. Ví dụ: `matrix=[[1,0],[0,1]]`.
* **N-Tensor**: Khi rank của Tensor tăng lên lớn hơn 2, chúng được gọi chung là N-Tensor.

> **Lưu ý**: Khái niệm về chiều trong Tensorflow và Python có sự sai khác lẫn nhau. Chiều trong python chính là bậc trong Tensorflow. Chiều trong Tensorflow là số lượng elements có trong bậc cuối cùng của Tensor tương ứng. Ví dụ `Tensor = [[[1,1,1],[2,3,4]]]` có chiều bằng 3, `Tensor = [[1,1,1],[2,3,4]]` vẫn có chiều bằng 3.

### 1.4. Shape
Shape là một tuple trong python có số chiều bằng với rank của Tensor tương ứng, dùng để mô tả lại cấu trúc của Tensor đó. Ví dụ:

* `Tensor = 1` sẽ có `Shape = ()`.
* `Tensor = [1]` sẽ có `Shape = (1)`.
* `Tensor = [[[1,1,1],[0,2,4]]]` sẽ có `Shape = (1,1,3)`.
* `Tensor = [[1,1,1],[0,2,4]]` sẽ có `Shape = (1,3)`.

### 1.5. Op
Operator được viết tắt là op, là toán tử được dùng để thực thi Tensor tại node đó. Các toán tử này có thể là Const (Hằng số), Variable (Biến số), Add (Phép cộng), Mul (Phép nhân)... 

### 1.6. DType
Đây là kiểu dữ liệu của các elements trong Tensor. Mỗi Tensor chỉ có duy nhất một thuộc tính DType nên tất cả các elements của tensor sẽ cùng kiểu dữ liệu.

## 2. Chương trình Hello World đầu tiên
Nào, bắt đầu code chương trình đầu tiên.
```python
import tensorflow as tf
hello = tf.constant('Hello World!')
sess = tf.Session()
print(sess.run (hello))
```
Kết quả trả về: `'Hello World!'`

Lý giải:
- Dòng 1: Import thư viện tensorflow vì trong python không có sẵn thư viện này.
- Dòng 2: Tạo một Node, Node này có tên là hello, Operator là constant, DType là tf.string, Shape là (). Để kiểm tra thông tin của nó, bạn có thể thêm dòng hello để biết thông tin của biến này.
- Dòng 3: Tạo một Session? Session được hiểu đơn giản là phiên làm việc, phải có session thì mới có thể thực thi operation của các node.
- Dòng 4: Session chạy thông qua phương thức run. Do đó, lệnh này để in ra kết quả của việc session run.

## 3. Xây dựng một graph đơn giản
Các bạn thử chạy chương trình sau và tự xem kết quả của mình để hiểu hơn nhé.
```python
import tensorflow as ts # import thư viện quen thuộc
node1 = tf.constant(1.0, tf.float32) #tạo node1 có kiểu tf.float32
node2 = tf.constant(2.0) # viết như này thì cũng ngầm mặc định hiểu là kiểu tf.float32
node3 = tf.add(node1, node2) # tạo node3 thông qua toán tử cộng từ 2 node trên
print("node1: ", node1, " node2: ", node2) # in thông tin của node1, node2
print("node3: ", node3) # in thông tin của node3 
sess = tf.Session() # tạo session
print("sess.run ([node1, node2]): ", sess.run ([node1, node2])) # in thông tin của session run
print("sess.run(node3): ", sess.run(node3)) # in thông tin của session run
```
## 4. Ứng dụng vào một bài toán tính tổng
```python
import tensorflow as tf

num_1 = tf.placeholder(tf.int32)
num_2 = tf.placeholder(tf.int32)
sumTwoNums = tf.add(num_1,num_2)

num1 = 1
num2 = 2

with tf.Session() as sess:
   result_sum = sess.run(sumTwoNums , feed_dict={num_1:num1,num_2:num2})

print(result_sum)
```
Ở đây mình có nhắc đến placeholder, nó là một khái niệm khá hay, thay vì bạn có thể truyền `num1 = 1, num2 = 2` thì bạn cũng có thể truyền `num1 = [1,2], num2 = [3,4]`. Bạn thử check và kiểm tra kết quả nhé.


