import tensorflow as tf


class polyval(object):
    def __init__(self, orders, d, coeffs, name=None):
        self.orders = orders
        self.d = d
        self.coeffs = coeffs

        self.input_dim = self.orders.shape[1]
        self.name = name

        self.declare_vars()

    def declare_vars(self):
        self.x = []
        for idxState in range(self.input_dim):
            self.x.append(
                tf.placeholder(tf.float32, shape=[None, 1], name='input')
            )
        self.y = 0.0
        self.poly()

    def poly(self):
        for order_item, coeff in zip(self.orders, self.coeffs):
            item = 1.0
            for idxState in range(self.input_dim):
                item *= (
                    self.x[idxState]**order_item[idxState] *
                    (1.0 - self.x[idxState])**(
                        self.d[idxState] - order_item[idxState]
                    )
                )
            self.y = tf.add(self.y, coeff * item)

    def __call__(self, sess, x_in):
        feed_dict = {}
        for idxState in range(self.input_dim):
            feed_dict.update(
                {self.x[idxState]: x_in[:, idxState].reshape(-1, 1)}
            )
        result = sess.run(self.y, feed_dict=feed_dict)

        return result
