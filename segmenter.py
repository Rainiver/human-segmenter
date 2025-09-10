# coding=utf-8
# date:
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import cv2
import os

class human_segmenter(object):
    def __init__(self, model_path,is_encrypted_model=False):
        super(human_segmenter, self).__init__()
        f = tf.gfile.FastGFile(model_path, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_graph = tf.import_graph_def(graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 占用GPU 30%的显存
        self.sess = tf.InteractiveSession(graph=persisted_graph, config=config)

        print("human_segmenter init done")
    
    def image_preprocess(self, img):
        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = img[:, :, ::-1]
        img = img.astype(np.float)
        return img
    
    def run(self, img):
        image_feed = self.image_preprocess(img)
        output_img_value, logits_value = self.sess.run([self.sess.graph.get_tensor_by_name("output_png:0"), self.sess.graph.get_tensor_by_name("if_person:0")],
                                                  feed_dict={self.sess.graph.get_tensor_by_name("input_image:0"): image_feed})
        mask = output_img_value[:, :, -1]
        output_img_value = cv2.cvtColor(output_img_value, cv2.COLOR_RGBA2BGRA)
        # return output_img_value
        return mask



if __name__ == "__main__":
    # img = cv2.imread('12345/images/0001.jpg')
    # print(img.shape)
    # fp = human_segmenter(model_path='assets/matting_human.pb')
    #
    # rgba = fp.run(img)
    # cv2.imwrite("res.png",rgba)
    # print("test done")
    fp = human_segmenter(model_path='matting_human.pb')
    directory_name = './data'
    for filename in sorted(os.listdir(directory_name)):
        # if int(filename.split('.')[-2].split('_')[-2]) > 0:
            # img = cv2.imread(directory_name + "/" + filename)
            # rgba = fp.run(img)
            # cv2.imwrite(f"./mask/{filename}", rgba)
            # print(f"./mask/{filename}")
        img = cv2.imread(directory_name + "/" + filename)
        rgba = fp.run(img)
        mask_name = filename.split('.')[-2] + '_mask.jpg'
        cv2.imwrite(f"./mask/{mask_name}", rgba)
        print(f"./mask/{mask_name}")
    print("test done")
