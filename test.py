import tensorflow as tf, time, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # less verbose
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)  # optional but helpful
    tf.debugging.set_log_device_placement(True)  # prints which device runs each op

    # small warmup
    x = tf.random.normal([1024, 1024])
    y = tf.random.normal([1024, 1024])
    with tf.device('/GPU:0'):
        _ = tf.matmul(x, y)

    # timing GPU vs CPU
    a = tf.random.normal([2048, 2048])
    b = tf.random.normal([2048, 2048])

    t0 = time.time()
    with tf.device('/GPU:0'):
        c = tf.matmul(a, b)
        _ = c.numpy()
    print("GPU matmul secs:", time.time() - t0)

    t0 = time.time()
    with tf.device('/CPU:0'):
        c = tf.matmul(a, b)
        _ = c.numpy()
    print("CPU matmul secs:", time.time() - t0)
else:
    print("No GPU detected by TensorFlow.")
