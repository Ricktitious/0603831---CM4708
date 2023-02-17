from Keras_OCR import process_single_file as Keras_OCR
from Textract_OCR import process_single_file as Textract_OCR
from Tesseract_OCR import process_single_file as Tesseract_OCR
import os

def ocr_comparison(file):
    if os.path.splitext(file)[1] != '.png':
        raise Exception('File must be a png file. File provided: ' + file)
    print('OCR Comparison')
    print('File: ' + file)
    print('Running Keras OCR on ' + file)
    keras_data, keras_image = Keras_OCR(file)
    print('Running Textract OCR on ' + file)
    textract_data, textract_image = Textract_OCR(file)
    print('Running Tesseract OCR on ' + file)
    tesseract_data, tesseract_image = Tesseract_OCR(file)
    print('OCR Comparison Complete')


if __name__ == '__main__':
    file = r'Data/MAPG-L-0010-040-D-AB00 - 000 - Z17.png'
    ocr_comparison(file)


