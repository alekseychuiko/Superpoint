# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:38:02 2019

@author: achuiko
"""
import model
import argparse
import cv2
import time
import extractor
import detector
import numpy as np
import videostreamer
from pathlib import Path
import tensorflow as tf

from settings import EXPER_PATH  # noqa: E402

def preprocess_image(img):
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img
    return img_preprocessed



if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('weights_name', type=str)
    parser.add_argument('--object_path', type=str,
        help='Path to object to detect')
    parser.add_argument('--H', type=int, default=720,
        help='Input image height (default: 720).')
    parser.add_argument('--W', type=int, default=1280,
        help='Input image width (default:1280).')
    parser.add_argument('--camid', type=int, default=0,
        help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
    parser.add_argument('--waitkey', type=int, default=1,
        help='OpenCV waitkey time in ms (default: 1).')
    parser.add_argument('--show_keypoints', type=int, default=0,
        help='0 - dont show keypoints, 1 - show matched keypoints, 2 - show all keypoints (default: not show)')
    parser.add_argument('--matcher_multiplier', type=float, default=0.8,
        help='Filter matches using the Lowes ratio test (default: 0.8).')
    parser.add_argument('--norm_type', type=int, default=1,
        help='0 - L1, 1 - L2, 2 - L2SQR, 3 - HAMMING, 4 - HAMMING (default: 1)')
    parser.add_argument('--method', type=int, default=0,
        help='0 - RANSAK, 1 - LMEDS, 2 - RHO (default: 0)')
    parser.add_argument('--repr_threshold', type=int, default=3,
        help='Maximum allowed reprojection error to treat a point pair as an inlier (used in the RANSAC and RHO methods only) (default: 3)')
    parser.add_argument('--max_iter', type=int, default=2000,
        help='Maximum number of RANSAC iterations (default: 2000)')
    parser.add_argument('--confidence', type=float, default=0.995,
        help='homography confidence level (default: 0.995).')
    parser.add_argument('--k_best', type=int, default=1000,
                        help='Maximum number of keypoints to keep \
                        (default: 1000)')
    opt = parser.parse_args()
    print(opt)
    
    norm_type = cv2.NORM_L1
    if opt.norm_type == 0 : norm_type = cv2.NORM_L1
    elif opt.norm_type == 1 : norm_type = cv2.NORM_L2
    elif opt.norm_type == 2 : norm_type = cv2.NORM_L2SQR
    elif opt.norm_type == 3 : norm_type = cv2.NORM_HAMMING
    else : norm_type = cv2.NORM_HAMMING2
      
    method = cv2.RANSAC
    if opt.method == 0 : method = cv2.RANSAC
    elif opt.method == 1 : method = cv2.LMEDS
    else : method = cv2.RHO    
    
    print('Load trained models...')
    weights_root_dir = Path(EXPER_PATH, 'saved_models')
    weights_root_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(weights_root_dir, opt.weights_name)

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(sess,
                                   [tf.saved_model.tag_constants.SERVING],
                                   str(weights_dir))

        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
        output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')
        
        objDetector = detector.Detector(opt.matcher_multiplier, norm_type, method, opt.repr_threshold, opt.max_iter, opt.confidence)
        
        vs = videostreamer.VideoStreamer("camera", opt.camid, opt.H, opt.W, 1, '')
        
        print('Running Demo.')
        
        win = 'SuperPoint Tracker'
        objwin = 'Object'
        cv2.namedWindow(win)
        cv2.namedWindow(objwin)
        
        obj = model.ModelFile(opt.object_path)
        greyObj = cv2.cvtColor(obj.image, cv2.COLOR_BGR2GRAY)
        img  = preprocess_image(greyObj)
        out = sess.run([output_prob_nms_tensor, output_desc_tensors],
                        feed_dict={input_img_tensor: np.expand_dims(img, 0)})
        keypoint_map = np.squeeze(out[0])
        descriptor_map = np.squeeze(out[1])
        kpObj, descObj = extractor.extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map, opt.k_best)
        if opt.show_keypoints != 0:
            objImg = cv2.drawKeypoints(greyObj, kpObj, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow(objwin, objImg)
        else:
            cv2.imshow(objwin, greyObj)
        
        while True:
            start = time.time()
    
            # Get a new image.
            greyImg, status = vs.next_frame()
        
            if status is False:
                break
    
            # Get points and descriptors.    
            img  = preprocess_image(greyImg)
            out = sess.run([output_prob_nms_tensor, output_desc_tensors],
                           feed_dict={input_img_tensor: np.expand_dims(img, 0)})
            keypoint_map = np.squeeze(out[0])
            descriptor_map = np.squeeze(out[1])
            kpImg, descImg = extractor.extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map, opt.k_best)
            out = objDetector.detect((np.dstack((greyImg, greyImg, greyImg))).astype('uint8'), kpObj, kpImg, descObj, descImg, obj, opt.show_keypoints)
        
            end1 = time.time()
            cv2.imshow(win, out)
            
            key = cv2.waitKey(opt.waitkey) & 0xFF
            if key == ord('q'):
                print('Quitting, \'q\' pressed.')
                break
        
            end = time.time()
            net_t = (1./ float(end1 - start))
            total_t = (1./ float(end - start))
            print('Processed image %d (net+post_process: %.2f FPS, total: %.2f FPS).' % (vs.i, net_t, total_t))
        
        
    cv2.destroyAllWindows()

    print('==> Finshed Demo.')