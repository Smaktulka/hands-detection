import tensorflow as tf

from convertion import SaveableModel


class CompleteDetectorTF(tf.keras.Model, SaveableModel):
    def get_model_input(self):
        input_loc = tf.keras.Input(shape=(896, 4), name='loc_data')
        input_conf = tf.keras.Input(shape=(896, 3), name='conf_data')
        input_dbox = tf.keras.Input(shape=(896, 4), name='dbox_list')
        return [input_loc, input_conf, input_dbox]

    def __init__(self, conf_thresh: float = 0.6, top_k: int = 200, nms_thresh: float = 0.6):
        super(CompleteDetectorTF, self).__init__()
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh

    @tf.function
    def decode(self, loc, dbox_list):
        boxes = tf.concat((
            dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, :2],
            dbox_list[:, 2:] * tf.exp(loc[:, 2:] * 0.2)), axis=1)

        boxes = tf.tensor_scatter_nd_update(boxes, tf.range(tf.shape(boxes)[0])[:, tf.newaxis],
                                            tf.concat((
                                                boxes[:, :2] - boxes[:, 2:] / 2,
                                                boxes[:, :2] - boxes[:, 2:] / 2 + boxes[:, 2:]), axis=1))

        return boxes

    @tf.function
    def call(self, loc_data: tf.Tensor, conf_data: tf.Tensor, dbox_list: tf.Tensor):
        num_batch = tf.shape(loc_data)[0]
        conf_data = tf.nn.softmax(conf_data)
        dbox_list = dbox_list[0]
        output = tf.TensorArray(tf.float32, size=num_batch)
        find_smth = False
        for i in tf.range(num_batch):
            decoded_boxes = self.decode(loc_data[i], dbox_list)
            conf_scores = conf_data[i]
            total_dets = 0
            classes_num = tf.shape(conf_scores)[1]
            for cl in tf.range(1, classes_num):
                c_mask = tf.greater(conf_scores[:, cl], self.conf_thresh)
                scores = conf_scores[:, cl][c_mask]
                if tf.size(scores) == 0:
                    continue

                l_mask = tf.expand_dims(c_mask, axis=-1)
                l_mask = tf.broadcast_to(l_mask, tf.shape(decoded_boxes))
                boxes = tf.boolean_mask(decoded_boxes, l_mask)
                boxes = tf.reshape(boxes, (-1, 4))

                ids = tf.image.non_max_suppression(boxes, scores, 3, 0.6)
                if total_dets + tf.size(ids) < self.top_k:
                    selected_scores = tf.expand_dims(tf.gather(scores, ids), axis=-1)
                    selected_boxes = tf.gather(boxes, ids)
                    class_tensor = tf.fill((tf.shape(ids)[0], 1), (cl - 1))
                    class_tensor = tf.cast(class_tensor, dtype=tf.float32)
                    combined = tf.concat([selected_scores, selected_boxes, class_tensor], axis=1)
                    output = output.write(i, combined)
                    find_smth = True
                    total_dets += tf.size(ids)

        if find_smth:
            output = output.stack()
            if len(output[0, :, 0] > 0.5) > 0:
                filtered_dets = tf.boolean_mask(output[0], output[0, :, 0] > 0.5)
                if tf.shape(filtered_dets)[0] > 0:
                    return filtered_dets
                else:
                    return tf.zeros(shape=(1, 6), dtype=tf.float32)
            else:
                return tf.zeros(shape=(1, 6), dtype=tf.float32)
        else:
            return tf.zeros(shape=(1, 6), dtype=tf.float32)


class OptimizedDetectorTF(tf.keras.Model, SaveableModel):
    def __init__(self, conf_thresh: float = 0.6):
        super(OptimizedDetectorTF, self).__init__()
        self.conf_thresh = conf_thresh

    def get_model_input(self):
        input_conf = tf.keras.Input(shape=(896, 3), name='conf_data')
        return [input_conf]

    @tf.function
    def call(self, conf_data: tf.Tensor):
        num_batch = tf.shape(conf_data)[0]
        classes_num = tf.shape(conf_data)[2]
        conf_data = tf.nn.softmax(conf_data)
        conf_preds = tf.transpose(conf_data, perm=[0, 2, 1])
        is_find = False
        class_for_ret = classes_num - 1
        for i in tf.range(num_batch):
            conf_scores = conf_preds[i]
            if is_find:
                break

            for cl in tf.range(1, classes_num):
                if is_find:
                    break

                if not tf.reduce_any(conf_scores[cl] > self.conf_thresh):
                    continue

                is_find = True
                class_for_ret = cl - 1

        return class_for_ret
