# -*- coding: utf-8 -*-

import collections
import json
import logging
import os
import time
import warnings
import pydub
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

from config import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

with open('model_labels.json', 'r') as labels_file:
    labels = json.load(labels_file)

signal = np.zeros((AUDIO_DURATION * SAMPLING_RATE, 1), dtype='float32')
spectrogram = np.zeros((MEL_BANDS, AUDIO_DURATION * SAMPLING_RATE // CHUNK_SIZE), dtype='float32')
audio_queue = collections.deque(maxlen=1000)  # Queue for incoming audio blocks
last_chunk = np.zeros((CHUNK_SIZE, 1), dtype='float32')  # Short term memory for the next step

predictions = np.zeros((len(labels), AUDIO_DURATION * SAMPLING_RATE // (BLOCK_SIZE * PREDICTION_STEP)), dtype='float32')
live_audio_feed = collections.deque(maxlen=1)

model = None


def get_raspberry_stats():
    freq = None
    temp = None
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as file:
            temp = int(file.read())
            temp /= 1000.
            temp = np.round(temp, 1)
            temp = '{}\'C'.format(temp)
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as file:
            freq = int(file.read())
            freq /= 1000.
            freq = '{} MHz'.format(int(freq))
    except:
        pass

    return temp, freq


def capture_audio(block, block_len, time, status):
    audio_queue.append(block.copy())


def classify(segments):
    X = np.stack(segments)
    X -= AUDIO_MEAN
    X /= AUDIO_STD
    pred = model.predict(X)
    pred = np.average(pred, axis=0, weights=np.arange(len(pred)) + 1)

    return pred


if __name__ == '__main__':
    # Import classifier model
    logger.info('Initializing a convolutional neural network model...')
    model

    THEANO_FLAGS = ('device=cpu,'
                    'floatX=float32,'
                    'dnn.conv.algo_bwd_filter=deterministic,'
                    'dnn.conv.algo_bwd_data=deterministic')

    os.environ['THEANO_FLAGS'] = THEANO_FLAGS
    os.environ['KERAS_BACKEND'] = 'theano'

    import keras

    keras.backend.set_image_dim_ordering('th')

    with open('model.json', 'r') as file:
        cfg = file.read()
        model = keras.models.model_from_json(cfg)

    model.load_weights('model.h5')
    logger.debug('Loaded Keras model with weights.')


    # for block in sf.blocks('dataset/audio/xx.wav', blocksize=BLOCK_SIZE, dtype='float32'):
    #     if count < 800:
    #         audio_queue.append(block.copy())
    #     count = count + 1

    blocks = []
    processing_queue = collections.deque()

    print(len(audio_queue))

    try:
        audio = pydub.AudioSegment.from_file('dataset/audio/d_1.wav').set_frame_rate(SAMPLING_RATE).set_channels(2)
        print(len(audio._data))
        audio = (np.fromstring(audio._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)
        # audio = audio.astype('float32')


        print(audio)

        # Populate spectrogram
        new_spec = librosa.feature.melspectrogram(audio,
                                              SAMPLING_RATE, n_fft=FFT_SIZE,
                                              hop_length=CHUNK_SIZE, n_mels=MEL_BANDS)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Ignore log10 zero division
            new_spec = librosa.core.perceptual_weighting(new_spec, MEL_FREQS, amin=1e-5,
                                                     ref_power=1e-5, top_db=None)
        new_spec = np.clip(new_spec, 0, 100)
        n_chunks = np.shape(new_spec)[1]
        spectrogram[:, :-n_chunks] = spectrogram[:, n_chunks:]
        spectrogram[:, -n_chunks:] = new_spec

        # Classify incoming audio
        predictions[:, :-1] = predictions[:, 1:]
        offset = SEGMENT_LENGTH // 2
        pred = classify([
            np.stack([spectrogram[:, -(SEGMENT_LENGTH + offset):-offset]]),
            np.stack([spectrogram[:, -SEGMENT_LENGTH:]]),
        ])

        predictions[:, -1] = pred
        print(pred)
        target = labels[np.argmax(pred)]

        # Final outcome Value
        print(pred[np.argmax(pred)])
        #  Final outcome Tag
        print(target)


    except Exception as e:
        print e



