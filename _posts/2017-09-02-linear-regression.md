---
title: Bài toán Linear Regression
author: huongnv75
date: 2017-09-02 12:00:00 +0700
categories: [Blogging, Tensorflow Tutorial]
tags: [tensorflow]
---
## 1. Tổng quan phương pháp
Bài toán này đơn giản được trình bày như sau:
* **Bước 1**: Biến những lý thuyết thực tế thành toán học, phải xác định được rõ sự phụ thuộc các biến để đưa ra đầu vào và đầu ra của bài toán.

$${y=\color{Red}f(x)}$$

* Khi đó,hàm $$\color{Red}f$$ trên chưa biết nhưng ta biết được chính xác đầu vào $$x$$ và đầu ra $$y$$ của bài toán ta đang thực hiện.
* **Bước 2**: Thu thập dữ liệu. Ở bước này, mục tiêu là ta đo được càng nhiều giá trị càng tốt. Tức là cứ ứng với mỗi $$x_{i}$$ ta thu được giá trị $$y_{i}$$ tương ứng. Giả sử rằng, ở bước này ta thu thập được $$m$$ cặp dữ liệu.

|STT | $$x$$ | $$y$$  |
|-------|--------|---------|
| $$1$$ | $$x_{1}$$ | $$y_{1}$$ |
| $$2$$ | $$x_{2}$$ | $$y_{2}$$ |
| $$3$$ | $$x_{3}$$ | $$y_{3}$$ |
| ... | ... | ... |
| $$m$$ | $$x_{m}$$ | $$y_{m}$$ |


* **Bước 3**: Nhiệm vụ của chúng ta là giờ có một giá trị $$x_{k}$$ nào đó, mà ta muốn tìm ra giá trị $$y_{k}$$ tương ứng. Ở đây, giải thuật sẽ tìm ra một hàm $$H(x)$$ (H là viết tắt của Hypothesis) theo dạng sau:

$$H(x)=Wx+b$$

* Đồng thời sẽ xác định thêm hàm mất mát phụ thuộc vào hai tham số $$W$$ (W là viết tắt của Weight) và $$b$$ (b là viết tắt của bias) như sau:

$$cost(W,b)=\frac{1}{m}\sum_{i=1}^{m}(H(x_{i})-y_{i})^2$$

* Mục tiêu là sẽ tìm $$W$$ và $$b$$ sao cho hàm $$cost(W,b)$$ có giá trị nhỏ nhất.  Khi đó, giá trị $$y_{k}$$ sẽ được tính đơn giản bằng công thức sau:

$$y_{k}=H(x_{k})$$

* Trong phạm vi bài viết này, mình sẽ implement thuật toán trên một cách đơn giản nhất mà chưa đào sâu cách tìm $$W$$ và $$b$$ cho bài toán.

## 2. Lập trình
### Dựng hàm Hypothesis
```python
# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# Our hypothesis XW+b
hypothesis = x_train * W + b
```
### Dựng hàm Cost
```python
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
```
### Dựng thuật toán tìm min
```python
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
```
### Cho session run
```python
# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2017):
   sess.run(train)
   if step % 20 == 0:
       print(step, sess.run(cost), sess.run(W), sess.run(b))
```
## 3. Chiêm ngưỡng sản phẩm
```python
import tensorflow as tf

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2017):
   sess.run(train)
   if step % 20 == 0:
       print(step, sess.run(cost), sess.run(W), sess.run(b))
```
Bạn có thể thay đổi một chút bằng cách sử dụng placeholder để truyền đầu vào tùy ý như [trong đây]({% post_url 2017-09-01-tensorflow %}) đã nhắc tới.
```python
import tensorflow as tf
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Our hypothesis XW+b
hypothesis = X * W + b
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line with new training data
for step in range(2017):
   cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
       feed_dict={X: [1, 2, 3, 4, 5, 6, 7], 
                  Y: [2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1]})
   if step % 20 == 0:
       print(step, cost_val, W_val, b_val)
```
Sau đó, để test nhanh một giá trị nào đó, bạn cần gõ thêm đoạn sau:
```python
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
```
