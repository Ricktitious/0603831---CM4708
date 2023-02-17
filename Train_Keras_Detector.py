import datetime
import string
import math
import os
import tensorflow as tf
import sklearn.model_selection
import cv2
import keras_ocr

"""
This script was taken from the keras-ocr github page and modified to work with the custom fonts and backgrounds.
The fonts used for technical drawings is based on the Roman font from the AutoCAD 2018 installation.
The font used in autocad is a .shx file, which is a font file that is used in AutoCAD.
The romans.fft file is a converted shx font
For the background, I took a P&ID CAD drawing and removed text from it.
Then converted them to png files.
"""

data_dir = '.'
# Custom alphabet to include some symbols that are commonly found in P&ID drawings
alphabet = string.digits + string.ascii_letters + '!?."#/\\'
recognizer_alphabet = ''.join(sorted(set(alphabet.lower())))

# load the fonts, we need multiple, but we're only using romans__.ttf.
fonts = [r"fonts\Romans_SHX\romans__.ttf"] * 1000

backgrounds = [os.path.join(r"backgrounds", file) for file in os.listdir(r"backgrounds")]
text_generator = keras_ocr.data_generation.get_text_generator(alphabet=alphabet)
# print('The first generated text is:', next(text_generator))

def get_train_val_test_split(arr):
    train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
    val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
    return train, val, test


background_splits = get_train_val_test_split(backgrounds)
font_splits = get_train_val_test_split(fonts)

image_generators = [
    keras_ocr.data_generation.get_image_generator(
        height=1024,
        width=1024,
        text_generator=text_generator,
        font_groups={
            alphabet: current_fonts
        },
        backgrounds=current_backgrounds,
        # font_size=(6, 20),
        # margin=20,
        # rotationX=(0, 30),
        # rotationY=(0, 30),
        # rotationZ=(0, 0),
        font_size=(20, 40),
        margin=50,
        rotationX=(-0.2, 0.2),
        rotationY=(-0.05, 0.05),
        rotationZ=(-15, 15)

    ) for current_fonts, current_backgrounds in zip(
        font_splits,
        background_splits
    )
]

# See what the first validation image looks like.
image, lines = next(image_generators[1])
text = keras_ocr.data_generation.convert_lines_to_paragraph(lines)
print('The first generated validation image (below) contains:', text)
cv2.imwrite('validation_image.png', image)

# load the detector
detector = keras_ocr.detection.Detector(weights='clovaai_general')
# load the recognizer
recognizer = keras_ocr.recognition.Recognizer(
    alphabet=recognizer_alphabet,
    weights='kurapan'
)
recognizer.compile()
# this makes only the last layer of the recogniser trainable
for layer in recognizer.backbone.layers:
    layer.trainable = False


detector_batch_size = 1
detector_basepath = os.path.join(data_dir, f'detector_{datetime.datetime.now().isoformat()}')
# create generators
detection_train_generator, detection_val_generator, detection_test_generator = [
    detector.get_batch_generator(
        image_generator=image_generator,
        batch_size=detector_batch_size
    ) for image_generator in image_generators
]
# train detector
detector.model.fit(
    detection_train_generator,
    steps_per_epoch=math.ceil(len(background_splits[0]) / detector_batch_size),
    epochs=10,
    workers=0,
    callbacks=[
        # tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=20),
        tf.keras.callbacks.CSVLogger(r'Models\Models_Detector.csv'),
        tf.keras.callbacks.ModelCheckpoint(filepath=r'Models\Models.h5')
    ],
    validation_data=detection_val_generator,
    validation_steps=math.ceil(len(background_splits[1]) / detector_batch_size),
    batch_size=detector_batch_size
)
print('Training Complete')

# save detector weights
detector.model.save_weights(r'Models\Detector_Weights.h5')

# load the trained detector into the pipeline
pipeline = keras_ocr.pipeline.Pipeline(detector=detector, recognizer=recognizer)
# get the next image from the generator
image, lines = next(image_generators[0])
# predict the text in the image
predictions = pipeline.recognize(images=[image])[0]
# draw the predictions on the image
drawn = keras_ocr.tools.drawBoxes(
    image=image, boxes=predictions, boxes_format='predictions'
)
# print the actual text and the predicted text
print(
    'Actual:', '\n'.join([' '.join([character for _, character in line]) for line in lines]),
    'Predicted:', [text for text, box in predictions]
)
# save the image
cv2.imwrite(r'Results\Detector_image.png', image)
# save the image with the predictions
cv2.imwrite(r'Results\Detector_Results.png', drawn)
print('Fin.')

