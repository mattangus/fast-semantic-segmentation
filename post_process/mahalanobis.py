import tensorflow as tf
import os
import numpy as np

from . import post_processor as pp
from third_party.doc_inherit import doc_inherit
from . import validation_metrics as metrics
from helpers import get_valid

class MahalProcessor(pp.PostProcessor):

    def __init__(self, model, outputs_dict, num_classes,
                    annot, image, path, ignore_label, process_annot,
                    num_gpus, batch_size,
                    #class specific
                    eval_dir, epsilon, global_cov, global_mean):
        super().__init__("Mahal", model, outputs_dict, num_gpus)
        self.eval_dir = eval_dir
        stats_dir = os.path.join(eval_dir, "stats")
        self.class_mean_file = os.path.join(stats_dir, "class_mean.npz")
        self.class_cov_file = os.path.join(stats_dir, "class_cov_inv.npz")
        self.num_classes = num_classes
        self.annot = annot
        self.image = image
        self.path = path
        self.mean_value = 508.7571
        self.std_value = 77.60572284853058
        self.epsilon = epsilon
        self.ignore_label = ignore_label
        self.global_cov = global_cov
        self.global_mean = global_mean
        self._process_annot = process_annot
        self._load_stats()
        self.batch_size = batch_size

        self.logit_gpu = "gpu:0"
        self.pre_process_gpu = "gpu:0"
        if num_gpus > 1:
            self.logit_gpu = "gpu:1"
            self.pre_process_gpu = "gpu:1"
        if num_gpus > 2:
            self.pre_process_gpu = "gpu:2"

    def _load_stats(self):
        print("loading means and covs")
        self.mean_v = np.load(self.class_mean_file)["arr_0"]
        self.var_inv_v = np.load(self.class_cov_file)["arr_0"]
        print("done loading")
        if self.global_cov:
            var_brod = np.ones_like(self.var_inv_v)
            self.var_inv_v = np.sum(self.var_inv_v, axis=(0,1,2), keepdims=True)*var_brod
        if self.global_mean:
            mean_brod = np.ones_like(self.mean_v)
            self.mean_v = np.mean(self.mean_v, axis=(0,1,2), keepdims=True)*mean_brod

    def _process_logits(self):
        final_logits = self.outputs_dict[self.model.final_logits_key]
        pred_shape = self.outputs_dict[self.model.main_class_predictions_key].shape.as_list()
        
        final_logits.set_shape([self.batch_size] + final_logits.shape.as_list()[1:])

        self.mean_p = tf.placeholder(tf.float32, self.mean_v.shape, "mean_p")
        self.var_inv_p = tf.placeholder(tf.float32, self.var_inv_v.shape, "var_inv_p")
        self.mean = tf.get_variable("mean", initializer=self.mean_p, trainable=False)
        self.var_inv = tf.get_variable("var_inv", initializer=self.var_inv_p, trainable=False)

        in_shape = final_logits.get_shape().as_list()

        mean_sub = tf.expand_dims(final_logits,-2) - self.mean
        mean_sub = tf.expand_dims(mean_sub, -2)

        self.pred = self.outputs_dict[self.model.main_class_predictions_key]

        tile_size = [in_shape[0]] + ([1] * (mean_sub._rank()-1))
        var_inv_tile = tf.tile(self.var_inv, tile_size)
        left = tf.matmul(mean_sub, var_inv_tile)
        mahal_dist = tf.squeeze(tf.sqrt(tf.matmul(left, mean_sub, transpose_b=True)))

        self.dist = mahal_dist

        self.img_dist = tf.expand_dims(tf.reshape(self.dist, in_shape[1:-1] + [self.num_classes]), 0)
        self.bad_pixel = tf.logical_or(tf.equal(self.img_dist, tf.zeros_like(self.img_dist)), tf.is_nan(self.img_dist))
        self.img_dist = tf.where(self.bad_pixel, tf.ones_like(self.img_dist)*float("inf"), self.img_dist)
        self.full_dist = tf.image.resize_bilinear(self.img_dist, (pred_shape[1],pred_shape[2]))
        self.dist_class = tf.expand_dims(tf.argmin(self.full_dist, -1),-1)
        self.min_dist = tf.reduce_min(self.full_dist, -1)

        min_dist_norm = (self.min_dist - self.mean_value) / self.std_value
        self.prediction = tf.nn.sigmoid(min_dist_norm)

    @doc_inherit
    def post_process_ops(self):
        main_pred = self.outputs_dict[self.model.main_class_predictions_key]
        self.weights = tf.to_float(get_valid(self.annot, self.ignore_label))

        with tf.device(self.logit_gpu):
            self._process_logits()
            self.annot_before = self.annot
            self.annot, self.num_classes = self._process_annot(self.annot, main_pred, self.num_classes)
            self.metrics, self.update = metrics.get_metric_ops(self.annot, self.prediction, self.weights)

    @doc_inherit
    def get_init_feed(self):
        ret = {self.mean_p: self.mean_v, self.var_inv_p: self.var_inv_v}
        return ret

    @doc_inherit
    def get_preprocessed(self):
        with tf.device(self.pre_process_gpu):
            if self.epsilon > 0.0:
                self.grads = tf.gradients(self.min_dist, self.image)
                return self.image - tf.squeeze(self.epsilon*tf.sign(self.grads), 1)
            return None

    @doc_inherit
    def get_vars_noload(self):
        return [self.mean, self.var_inv]

    @doc_inherit
    def get_fetch_dict(self):
        return [{"update": self.update}, {"metrics": self.metrics, "path": self.path, "img": self.prediction, "pred": self.pred, "annot": self.annot_before, "input": self.image}]

    @doc_inherit
    def get_feed_dict(self):
        return {}

    @doc_inherit
    def post_process(self, numpy_dict):
        results = metrics.get_metric_values(numpy_dict["metrics"])

        import matplotlib.pyplot as plt
        from skimage.segmentation import mark_boundaries
        from skimage.segmentation import slic

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim

        import cv2

        img = numpy_dict["input"][0].astype(np.uint8)
        uncert = numpy_dict["img"][0,...]
        pred = numpy_dict["pred"][0,...,0]
        annot = numpy_dict["annot"][0,...,0]
        path = numpy_dict["path"][0].decode()
        depth = cv2.imread(path.replace("leftImg8bit", "disparity"), cv2.IMREAD_UNCHANGED)

        segs = slic(uncert.astype(np.float64), n_segments = 1500, compactness=0.05, sigma = 0.01)
        
        class_edge = cv2.Canny(pred.astype(np.uint8), 0,0)
        class_dtform = cv2.distanceTransform(255 - class_edge, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
        border_probs = 1/np.sqrt(class_dtform)
        border_probs[class_dtform == 0] = np.nanmax(border_probs)
        border_probs[class_dtform == 0] = 1
        border_probs *= 0.99


        # class Net(nn.Module):
        #     def __init__(self):
        #         super(Net, self).__init__()
        #         self.conv1 = nn.Conv2d(1, 1, 7, padding=3)
        #         self.relu = nn.ReLU()
        #         self.conv2 = nn.Conv2d(1,1,7,padding=3)
                

        #     def forward(self, x):
        #         x = self.conv1(x)
        #         # x = self.relu(x)
        #         # x = self.conv2(x)

        #         return x

        # net = Net()
        # loss_fn = nn.MSELoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

        # ed = np.expand_dims

        # for i in range(10):
        #         # get the inputs

        #         inputs = torch.Tensor(np.transpose(ed(ed(uncert,0),-1), [0, 3, 1, 2]))
        #         labels = torch.Tensor(np.transpose(ed(ed(border_probs,0),-1), [0, 3, 1, 2]))

        #         # zero the parameter gradients
        #         optimizer.zero_grad()

        #         # forward + backward + optimize
        #         outputs = net(inputs)
        #         loss = loss_fn(outputs, labels)
        #         loss.backward()
        #         optimizer.step()

        #         # print statistics
        #         running_loss = loss.item()
        #         print('[%5d] loss: %.3f' % (i + 1, running_loss))

        # res = torch.sigmoid(net(inputs)).detach().numpy()
        # plt.imshow(res[0,0])
        # plt.show()

        # plt.imshow(mark_boundaries(uncert.astype(np.float64), segs))
        # plt.show()

        # for v in np.unique(segs):
        #     inds = np.transpose(np.nonzero(segs == v))

        #     mean = np.mean()

        #     import pdb; pdb.set_trace()

        import pdb; pdb.set_trace()

        # import matplotlib.pyplot as plt
        # import cv2
        # img = numpy_dict["img"][0]
        # pred = numpy_dict["pred"][0,...,0]
        # annot = numpy_dict["annot"][0,...,0]
        # input = numpy_dict["input"][0]
        # path = numpy_dict["path"]
        # kernel = np.ones((3,3),np.uint8)
        # class_edge = cv2.Canny(pred.astype(np.uint8), 0,0)
        # #class_dilated = cv2.dilate(edge, kernel, iterations=1)/255

        # def disc(n):
        #     y,x = np.ogrid[-n: n+1, -n: n+1]
        #     mask = x**2+y**2 <= n**2
        #     return mask.astype(np.uint8)

        # class_dtform = cv2.distanceTransform(255 - class_edge, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
        # border_probs = 1/np.sqrt(class_dtform)
        # border_probs[class_dtform == 0] = np.nanmax(border_probs)
        # border_probs[class_dtform == 0] = 1
        # border_probs *= 0.99

        # new_probs = cv2.dilate(cv2.erode(img, disc(7)), disc(7))*img

        # # conf_edge = cv2.Canny((img*255).astype(np.uint8),100,100,L2gradient=True)
        # # conf_dtform = cv2.distanceTransform(255 - conf_edge, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
        # # conf_dtform[conf_dtform > 10] = 10
        # # conf_norm_dtform = conf_dtform / 10

        # # dtform_pred = img*norm_dtform

        # pred[0,0] = 20
        # annot[annot==255] = 20

        # plt.subplot(2,2,1); plt.imshow(new_probs); plt.subplot(2,2,2); plt.imshow(pred); plt.subplot(2,2,3); plt.imshow(annot); plt.subplot(2,2,4); plt.imshow(input/255); 
        # plt.show()

        # import pdb; pdb.set_trace()

        return results


        return results
    
    @doc_inherit
    def get_output_image(self):
        return self.prediction
    
    @doc_inherit
    def get_prediction(self):
        return self.outputs_dict[self.model.main_class_predictions_key]
    
    @doc_inherit
    def get_weights(self):
        return self.weights
        