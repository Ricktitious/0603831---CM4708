import pytesseract
import pandas as pd
import cv2
import os

# pytesseract requires Tesseract to be installed
# this is the default location for Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def convert_df_to_boundingbox_dict(df):
    """
    Convert dataframe to dictionary of words and their coordinates
    :param df:
    :return: dictionary of words and their coordinates
    """
    # create dictionary of words and their coordinates
    boundingbox_dict = {}
    for index, row in df.iterrows():
        text = row['text']
        x1, y1, x2, y2 = row['left'], row['top'], row['left'] + row['width'], row['top'] + row['height']
        boundingbox_dict[index] = {'text': text, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    return boundingbox_dict

def annotate_image_with_boundingboxes(image, boundingbox_dict, with_text=False):
    """
    Annotate an image with bounding boxes
    :param image:
    :param boundingbox_dict:
    :param with_text:
    :return: annotated image
    """
    for key, value in boundingbox_dict.items():
        text = value['text']
        pt1 = (value['x1'], value['y1'])
        pt2 = (value['x2'], value['y2'])
        cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=3)
        if with_text:
            cv2.putText(image, text, pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image

def process_single_file(file):
    # Read image
    image = cv2.imread(file)
    # get wordblocks from image
    boxes = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    #convert to dataframe
    df = pd.DataFrame(boxes)
    # remove rows with no text and text is trimmed
    df = df[(df['conf'] != -1) & (df['text'].str.strip() != '')]
    # convert to dictionary
    boundingbox_dict = convert_df_to_boundingbox_dict(df)
    # annotate image with bounding boxes
    image2 = annotate_image_with_boundingboxes(image, boundingbox_dict)
    # path handling
    dir, filename = os.path.split(file)
    fname, ext = os.path.splitext(filename)
    outfile = fname + '_Tesseract.png'
    outfile = os.path.join('Results', outfile)
    # save image
    cv2.imwrite(outfile, image2)
    return boundingbox_dict, image2

