# Weight-Agnostic-Networks
Tensorflow implementation of weight agnostic networks as described in Gaier et. al

Common Issues:
pip
  1) Out of space
  Use pip install --cache-dir=./{{swap_folder}} --build ./f{{swap_folder}} -r requirements.txt
  2) B1as GEMM Failed to launch (tensorflow-gpu specific):
  Hop over to tensorflow_backend.py (in the keras library) and add these lines
  if _SESSION is None:
            if not os.environ.get('OMP_NUM_THREADS'):
                config = tf.ConfigProto(allow_soft_placement=True)
Add this line >>> config.gpu_options.allow_growth=True
            else:
                num_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = tf.ConfigProto(intra_op_parallelism_threads=num_thread,
                                        allow_soft_placement=True)
Add this line >>> config.gpu_options.allow_growth=True
            _SESSION = tf.Session(config=config)
        session = _SESSION
