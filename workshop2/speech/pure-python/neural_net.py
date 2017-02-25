# import tensorflow
import tensorflow as tf
# import the corpus parser
import corpus_parser as corpus

# dimension of prediction vector
# dimension of input vector
input_dim, output_dim, num_stimuli = corpus.parse_files()

# start an interactive session
#
# With interactive sessions, statements that build the computation
# graph can be mixed in with statements that run the graph
sess = tf.InteractiveSession()

# tensorflow placeholder
x = tf.placeholder(tf.float32, shape=[None, input_dim])
y_ = tf.placeholder(tf.float32, shape=[None, output_dim])

# 'Variable' types are used for learned parameters
# weight matrix variable
W = tf.Variable(tf.zeros([input_dim, output_dim]))
# bias node variable
b = tf.Variable(tf.zeros([output_dim]))

# initializing all the parameters as zero is a bad idea.
#
# Use global_variables_initializer() to choose a smarter starting
# point

# define the procedure for computing a predicted value
y = tf.matmul(x, W) + b

# Loss function
#
# this defines how bad a single prediction is
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# Define learning rate
learning_rate = 0.01

# Define how parameters are adjusted
train_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(cross_entropy)

# Train the model
def validate():
    num_correct = 0
    for j in range(num_stimuli):
        corpus.make_sets()
        sess.run(tf.global_variables_initializer())
        print('epoch ',j)
        for i in range(1000):
            batch = corpus.next_stimulus()
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        test_set = corpus.test_stimulus()
        num_correct = num_correct + accuracy.eval(feed_dict={x: test_set[0], y_: test_set[1]})
    print(num_correct/num_stimuli)

def train():
    corpus.full_set()
    sess.run(tf.global_variables_initializer())
    for i in range(500):
        batch = corpus.next_stimulus()
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

def predict(vector):
    prediction = tf.argmax(y, 1)
    return prediction.eval(feed_dict={x: vector})[0]

def response(raw):
    sentence = raw.split()
    input_vec = corpus.vectorize(sentence)
    return predict(input_vec)
