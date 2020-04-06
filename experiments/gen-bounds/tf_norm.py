def sharpness_sigma(
    model_fn, image_tensor, label_tensor, training_accuracy, target_deviate,
    checkpoint_directory, upper=5., lower=0., search_depth=20, mtc_iter=15,
    ascent_step=20, num_batch=10, deviat_eps=1e-2, bound_eps=5e-3):
    sess = tf.Session()
    logits = model_fn(image_tensor)
    model_variables = model_fn.variables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(model_variables)
    saver.restore(sess, checkpoint_directory)

    perturb_ph, perturb_add, perturb_sub = add_noise_to_variables(model_variables)
    perturb_ph_list = [perturb_ph[k] for k in perturb_ph]
    original_weight = [sess.run(v) for v in model_variables]
    restore_original_weight_op = [
      tf.assign(v, vv) for v, vv in zip(model_variables, original_weight)
    ]

    perturbation = []
    for v, vo in zip(model_variables, original_weight):
    perturbation.append(v - vo)
    flattened_difference = flatten_and_concat(perturbation)
     perturb_norm = tf.norm(flattened_difference)

    projection_op = []
    target_norm = tf.placeholder(tf.float32, ())
    for v, vo in zip(model_variables, original_weight):
    projection_op.append(
        tf.assign(v, vo + target_norm / perturb_norm * (v - vo)))

    xent = tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=label_tensor)
    op = tf.train.GradientDescentOptimizer(learning_rate).minimize(-xent)
    correctly_predicted = tf.equal(
      tf.argmax(logits, axis=-1), tf.argmax(label_tensor, axis=-1))
    loss = tf.reduce_mean(tf.cast(correctly_predicted, tf.float32))

    h, l = upper, lower
    for j in range(search_depth):
    m = (h + l) / 2.
    min_accuracy = 10.
    for i in range(mtc_iter):
      sess.run(restore_original_weight_op)
      fd = get_gaussian_noise_feed_dict(perturb_ph, m)
      sess.run(perturb_add)
      for k in range(ascent_step):
        sess.run(op)
        new_norm = sess.run(perturb_norm)
        if new_norm > m: sess.run(projection_op, {target_norm: m})
        if j % 10 == 0:
          estimates = []
          for _ in range(num_batch):
            estimates.append(sess.run(loss))
          min_accuracy = min(min_accuracy, np.mean(estimates))
    deviate = abs(min_accuracy - training_accuracy)
    if h - l < bound_eps or abs(deviate - target_loss) < deviat_eps:
      return m
    if deviate > target_loss:
      h = m
    else:
      l = m


def add_noise_to_variables(variables):
  """Create tf ops for adding noise to a list of variables."""
  perturbation_ph = {}
  add_perturbation_op = []
  subtract_perturbation_op = []
  for v in variables:
    perturbation_ph[v] = tf.placeholder(
        tf.float32, shape=v.get_shape().as_list())
    add_perturbation_op.append(tf.assign_add(v, perturbation_ph[v]))
    subtract_perturbation_op.append(tf.assign_add(v, -perturbation_ph[v]))
  return perturbation_ph, add_perturbation_op, subtract_perturbation_op


def get_gaussian_noise_feed_dict(ph_list, scale):
  """Get noise with standard deviation of scale."""
  feed_dict = {}
  for ph in ph_list:
    feed_dict[ph] = np.random.normal(
        scale=scale, size=ph.get_shape().as_list())
  return feed_dict


def flatten_and_concat(variable_list):
  variable_list = [tf.reshape(v, [-1]) for v in variable_list]
  return tf.concat(variable_list, axis=0)


def norm_of_weights(weights):
  flat_weights = [np.reshape(w, -1) for w in weights]
  concat_weight = np.concatenate(flat_weights)
  weight_norm = np.linalg.norm(concat_weight)
  return weight_norm
