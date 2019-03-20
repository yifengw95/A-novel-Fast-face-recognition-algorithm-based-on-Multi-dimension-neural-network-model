import numpy as np
import tensorflow as tf
from time import time
import math
import argparse
from include.data import get_data_set, change_num_classes, get_mbatch
from include.model import model, lr
import pdb

def get_filename(aug_list, aug_prob, args):
    fname = "lr" + str(args.lr_mode) + "_"
    for c in range(len(aug_list)):
        fname += aug_list[c][0]
        if len(aug_list[c]) == 2:
            fname += str(aug_list[c][1])
        fname += ":" + str(aug_prob[c]) + "_"
    fname += args.opt + "_" + str(args.a0) + ".log"
    return fname

def convert_str2list(str_list, str_prob):
    str_list = str_list.split('-')
    aug_list = []
    for temp in str_list:
        temp = temp[1:-1]
        temp = temp.split(',')
        if len(temp) == 1:
            aug_list.append([temp[0]])
        else:   
            aug_list.append([temp[0], int(temp[1])])

    aug_prob = []
    for p in str_prob[1:-1].split(','): 
        aug_prob.append(float(p))

    return aug_list, aug_prob

p = argparse.ArgumentParser()
p.add_argument('-aug_list','--aug_list', help='aug list', required=True)
p.add_argument('-aug_prob','--aug_prob', help='aug prob', required=True)
p.add_argument('-opt','--opt', help='optimizer', required=False, type=str, default="sgd")
p.add_argument('-a0','--a0', help='alpha_0 for opt', required=False, type=float, default=0.1)
p.add_argument('-epoch','--epoch', help='epoch', required=False, type=int, default=60)
p.add_argument('-lr_mode', '--lr_mode', help='lr mode', required=False, type=int, default=0)
args = p.parse_args()


# PARAMS
_BATCH_SIZE = 128
_EPOCH = args.epoch
num_classes = 100
_SAVE_PATH = "./tensorboard/cifar-" + str(num_classes) + "-v1.0.0/"


aug_list, aug_prob = convert_str2list(args.aug_list, args.aug_prob)
log_filename = get_filename(aug_list, aug_prob, args)

outfile = open("./logs/" + log_filename, "w+")
outfile.write(str(aug_list) + "\n" + str(aug_prob) + "\n\n")
outfile.write("opt: " + args.opt + "\n" + "a0: " + str(args.a0) + "\n")


# GENERATORS
gen_train = get_mbatch("train", _BATCH_SIZE, num_classes, aug_list, aug_prob)
test_x, test_y = get_data_set("test")
test_x, test_y = change_num_classes(test_x, test_y, num_classes)
tf.set_random_seed(21)

# MODEL
x, y, logits, softmax, y_pred_cls, global_step, learning_rate = model(num_classes)

global_accuracy = 0
epoch_start = 0


#############################################################################################
# LOSS AND OPTIMIZER
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

# EE599: How to select optimization technique and details:
if args.opt == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
elif args.opt == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-08).minimize(loss, global_step=global_step)
################################################################################################


# PREDICTION AND ACCURACY CALCULATION
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# SAVER
merged = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=10)


# sess = tf.Session()
sess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    # gpu_options=tf.GPUOptions(allow_growth=True),
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
    )
)

train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)


# try:
#     print("\nTrying to restore last checkpoint ...")
#     last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
#     saver.restore(sess, save_path=last_chk_path)
#     print("Restored checkpoint from:", last_chk_path)
# except ValueError:
#     print("\nFailed to restore checkpoint. Initializing variables instead.")
#     sess.run(tf.global_variables_initializer())

# Saeed changed this for 599
sess.run(tf.global_variables_initializer())

def train(epoch):
    global epoch_start
    epoch_start = time()
    # TODO: 
    batch_size = int(50000/_BATCH_SIZE)
    i_global = 0

    for s in range(batch_size):
        batch_xs, batch_ys = gen_train.next()

        start_time = time()

        ######################################################################
        # EE599: Running the optimization on NN for one step: 
        i_global, _, batch_loss, batch_acc, y_pred_label, _logits, _softmax = sess.run(
            [global_step, optimizer, loss, accuracy, y_pred_cls, logits, softmax],
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch, args.a0, args.lr_mode)})
        ########################################################################


        duration = time() - start_time

        if s % 10 == 0:
            percentage = int(round((float(s)/batch_size)*100))
            
            bar_len = 29
            filled_len = int((bar_len*int(percentage))/100)
            bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)

            msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f}"
            txt = msg.format(i_global, bar, percentage, batch_acc, batch_loss)
            print(txt)
            outfile.write(txt + "\n")

    test_and_save(i_global, epoch)


def test_and_save(_global_step, epoch):
    global global_accuracy
    global epoch_start

    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(
            y_pred_cls,
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch, 0.1)}
        )
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()

    hours, rem = divmod(time() - epoch_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{})" 
    txt = mes.format((epoch+1), acc, correct_numbers, len(test_x))
    print(txt)
    outfile.write(txt + "\n")

    if global_accuracy != 0 and global_accuracy < acc:

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
        ])
        train_writer.add_summary(summary, _global_step)

        saver.save(sess, save_path=_SAVE_PATH + 'newModel', global_step=_global_step)

        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        txt = mes.format(acc, global_accuracy)
        print(txt)
        outfile.write(txt + "\n")
        global_accuracy = acc

    elif global_accuracy == 0:
        global_accuracy = acc

    print("##########################################################################")


def main():
    train_start = time()

    for i in range(_EPOCH):
        print("\nEpoch: {}/{}\n".format((i+1), _EPOCH))
        train(i)

    hours, rem = divmod(time() - train_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "Best accuracy pre session: {:.2f}, time: {:0>2}:{:0>2}:{:05.2f}"
    txt = mes.format(global_accuracy, int(hours), int(minutes), seconds)
    print(txt)
    outfile.write(txt + "\n")


if __name__ == "__main__":
    main()


sess.close()
