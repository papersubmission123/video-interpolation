import tensorflow as tf
import numpy as np
from scipy.ndimage import imread
from scipy import misc
from glob import glob
import os

import constants as c
from tfutils import log10

##
# Data
##

def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    new_frames /= (255 / 2)
    new_frames -= 1

    return new_frames

def denormalize_frames(frames):
    """
    Performs the inverse operation of normalize_frames.

    @param frames: A numpy array. The frames to be converted.

    @return: The denormalized frames.
    """
    new_frames = frames + 1
    new_frames *= (255 / 2)
    # noinspection PyUnresolvedReferences
    new_frames = new_frames.astype(np.uint8)

    return new_frames

def clip_l2_diff(clip):
    """
    @param clip: A numpy array of shape [c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
    @return: The sum of l2 differences between the frame pixels of each sequential pair of frames.
    """
    diff = 0
    for i in xrange(c.HIST_LEN):
        frame = clip[:, :, 3 * i:3 * (i + 1)]
        next_frame = clip[:, :, 3 * (i + 1):3 * (i + 2)]
        # noinspection PyTypeChecker
        diff += np.sum(np.square(next_frame - frame))

    return diff

def get_full_clips(data_dir, num_clips, num_rec_out=1):
    """
    Loads a batch of random clips from the unprocessed train or test data.

    @param data_dir: The directory of the data to read. Should be either c.TRAIN_DIR or c.TEST_DIR.
    @param num_clips: The number of clips to read.
    @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                        using the previously-generated frames as input. Default = 1.

    @return: An array of shape
             [num_clips, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + num_rec_out))].
             A batch of frame sequences with values normalized in range [-1, 1].
    """
    clips = np.empty([num_clips,
                      c.FULL_HEIGHT,
                      c.FULL_WIDTH,
                      (3 * (c.HIST_LEN + num_rec_out))])

    # get num_clips random episodes
    ep_dirs = np.random.choice(glob(os.path.join(data_dir, '*')), num_clips)
    #print(ep_dirs)

    # get a random clip of length HIST_LEN + num_rec_out from each episode
    for clip_num, ep_dir in enumerate(ep_dirs):
        ep_frame_paths = sorted(glob(os.path.join(ep_dir, '*')))
        #print(ep_frame_paths)
        file_list = ['pred_1.png', 'pred_2.png', 'pred_3.png', 'pred_4.png', 'target_1.png', 'target_2.png', 'target_3.png']
        i = 0
        for f in file_list:
            #frame_path = ep_dir + '/input00' + str(i+start_index+1) + '.jpg'
            frame_path = ep_dir + '/' + f
            frame = imread(frame_path, mode='RGB')
            frame = misc.imresize(frame, (c.FULL_HEIGHT, c.FULL_WIDTH))
            norm_frame = normalize_frames(frame)
            clips[clip_num, :, :, i * 3:(i + 1) * 3] = norm_frame
            i += 1
        """
        if (len(ep_frame_paths) == 0):
            continue
        start_index = np.random.choice(len(ep_frame_paths) - (c.HIST_LEN + num_rec_out - 1))
        clip_frame_paths = ep_frame_paths[start_index:start_index + (c.HIST_LEN + num_rec_out)]

        for i in range(0, (c.HIST_LEN+num_rec_out)):
            #frame_path = ep_dir + '/input00' + str(i+start_index+1) + '.jpg'
            frame_path = ep_dir + '/frame_' + str(i+start_index) + '.png'
            frame = imread(frame_path, mode='RGB')
            frame = misc.imresize(frame, (c.FULL_HEIGHT, c.FULL_WIDTH))
            norm_frame = normalize_frames(frame)
            clips[clip_num, :, :, i * 3:(i + 1) * 3] = norm_frame
        """

    return clips

def process_clip():
    """
    Gets a clip from the train dataset, cropped randomly to c.TRAIN_HEIGHT x c.TRAIN_WIDTH.

    @return: An array of shape [c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
             A frame sequence with values normalized in range [-1, 1].
    """
    clip = get_full_clips(c.TRAIN_DIR, 1, 4)[0]

    # Randomly crop the clip. With 0.05 probability, take the first crop offered, otherwise,
    # repeat until we have a clip with movement in it.
    take_first = np.random.choice(2, p=[0.95, 0.05])
    cropped_clip = np.empty([c.TRAIN_HEIGHT, c.TRAIN_WIDTH, 3 * (c.HIST_LEN + 1)])
    for i in xrange(100):  # cap at 100 trials in case the clip has no movement anywhere
        crop_x = np.random.choice(c.FULL_WIDTH - c.TRAIN_WIDTH + 1)
        crop_y = np.random.choice(c.FULL_HEIGHT - c.TRAIN_HEIGHT + 1)
        cropped_clip = clip[crop_y:crop_y + c.TRAIN_HEIGHT, crop_x:crop_x + c.TRAIN_WIDTH, :]

        if take_first or clip_l2_diff(cropped_clip) > c.MOVEMENT_THRESHOLD:
            break

    return cropped_clip

def get_train_batch():
    """
    Loads c.BATCH_SIZE clips from the database of preprocessed training clips.

    @return: An array of shape
            [c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
    """
    clips = np.empty([c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + c.PRED_LEN))],
                     dtype=np.float32)
    for i in xrange(c.BATCH_SIZE):
        #path = c.TRAIN_DIR_CLIPS + 'Clips' + str(np.random.choice(499999)) + '.npz'
        path = c.TRAIN_DIR_CLIPS  + str(np.random.choice(15000)) + '.npz'
        clip = np.load(path)['arr_0']

        """
        re_clip = np.empty([c.TRAIN_HEIGHT, c.TRAIN_WIDTH, 27])
        for j in range(0, 27, 3):
            re_clip[:, :, j:j+3] = misc.imresize(clip[:, :, j:j+3], (c.TRAIN_HEIGHT, c.TRAIN_WIDTH))
        clips[i] = re_clip
        """
        clips[i] = clip[:, :, :3*(c.HIST_LEN+c.PRED_LEN)]

    return clips

'''
def get_test_batch(test_batch_size, num_rec_out=1):
    """
    Test for 16 frames
    """
    num_clips = test_batch_size
    clips = np.empty([num_clips,
                      c.FULL_HEIGHT,
                      c.FULL_WIDTH,
                      (3 * 16)])

    # get num_clips random episodes
    ep_dirs = np.random.choice(glob(os.path.join(c.TEST_DIR, '*')), num_clips)
    #print(ep_dirs)

    # get a random clip of length HIST_LEN + num_rec_out from each episode
    for clip_num, ep_dir in enumerate(ep_dirs):
        ep_frame_paths = sorted(glob(os.path.join(ep_dir, '*')))
        start_index = np.random.choice(len(ep_frame_paths) - (c.HIST_LEN + num_rec_out - 1))
        clip_frame_paths = ep_frame_paths[start_index:start_index + (c.HIST_LEN + num_rec_out)]

        for i in range(0, (c.HIST_LEN+num_rec_out)):
            #frame_path = ep_dir + '/input00' + str(i+start_index+1) + '.jpg'
            frame_path = ep_dir + '/frame_' + str(i+start_index) + '.png'
            frame = imread(frame_path, mode='RGB')
            frame = misc.imresize(frame, (c.FULL_HEIGHT, c.FULL_WIDTH))
            norm_frame = normalize_frames(frame)
            clips[clip_num, :, :, i * 3:(i + 1) * 3] = norm_frame
    return clips
'''

def get_test_batch(test_batch_size, num_rec_out=1):
    """
    Test batch from npz files
    """

    clips = np.empty([c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + c.PRED_LEN))],
                     dtype=np.float32)
    st = np.random.choice(19000) + 80000
    for i in xrange(c.BATCH_SIZE):
        #path = c.TRAIN_DIR_CLIPS + 'Clips' + str(np.random.choice(499999)) + '.npz'
        #path = c.TEST_DIR  + str(np.random.choice(5000)+15000) + '.npz'
        path = c.TEST_DIR  + str(st+i) + '.npz'
        clip = np.load(path)['arr_0']

        clips[i] = clip[:, :, :3*(c.HIST_LEN+c.PRED_LEN)]
    return clips


##
# Error calculation
##

# TODO: Add SSIM error http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf
# TODO: Unit test error functions.

def psnr_error(gen_frames, gt_frames):
    """
    Computes the Peak Signal to Noise Ratio error between the generated images and the ground
    truth images.

    @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the
                       generator model.
    @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                      each frame in gen_frames.

    @return: A scalar tensor. The mean Peak Signal to Noise Ratio error over each frame in the
             batch.
    """
    shape = tf.shape(gen_frames)
    num_pixels = tf.to_float(shape[1] * shape[2] * shape[3])
    square_diff = tf.square(gt_frames - gen_frames)

    batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.reduce_sum(square_diff, [1, 2, 3])))
    return tf.reduce_mean(batch_errors)

def sharp_diff_error(gen_frames, gt_frames):
    """
    Computes the Sharpness Difference error between the generated images and the ground truth
    images.

    @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the
                       generator model.
    @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                      each frame in gen_frames.

    @return: A scalar tensor. The Sharpness Difference error over each frame in the batch.
    """
    shape = tf.shape(gen_frames)
    num_pixels = tf.to_float(shape[1] * shape[2] * shape[3])

    # gradient difference
    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    # TODO: Could this be simplified with one filter [[-1, 2], [0, -1]]?
    pos = tf.constant(np.identity(3*c.PRED_LEN), dtype=tf.float32)
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(gen_frames, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(gen_frames, filter_y, strides, padding=padding))
    gt_dx = tf.abs(tf.nn.conv2d(gt_frames, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(gt_frames, filter_y, strides, padding=padding))

    gen_grad_sum = gen_dx + gen_dy
    gt_grad_sum = gt_dx + gt_dy

    grad_diff = tf.abs(gt_grad_sum - gen_grad_sum)

    batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.reduce_sum(grad_diff, [1, 2, 3])))
    return tf.reduce_mean(batch_errors)
