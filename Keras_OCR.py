import pdf2image
import keras_ocr
import cv2
import os

# pdf2image requires poppler to be installed
# you can set POPPLER_PATH to the location of the bin folder
# this is the default location for poppler-0.68.0
POPPLER_PATH = r'C:\Program Files\poppler-0.68.0\bin'

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

def get_files_from_folder(folder):
    """Get all pdf files from a folder
    Args:
    folder (str): The folder to get the files from
    return: A list of files
    """

    files = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            files.append(os.path.join(folder, file))
    return files

def convert_pdfs_to_images(files, output_folder):
    """Convert pdf files to images
    Args:
    files (list): A list of pdf files
    output_folder (str): The folder to save the images to
    return: A list of dictionaries with the pdf file and the pages
    """
    new_files = []
    for file in files:
        file_dict = {
            "pdf": file,
            "pages": {}
        }
        pages = pdf2image.convert_from_path(file, 200, poppler_path=POPPLER_PATH)
        for i, page in enumerate(pages):
            output_name = os.path.join(output_folder, f'{os.path.basename(file)}_{i}.png')
            file_dict["pages"][i] = output_name
            page.save(output_name, 'PNG')
        new_files.append(file_dict)
    return new_files

def convert_images_to_keras_ocr(images):
    """Convert images to keras-ocr format
    Args:
    images (dict): A dictionary of images
    return: A list of images
    """
    images1 = []
    for i, image in images.items():
        images1.append(keras_ocr.tools.read(image))
    return images1

def get_prediction_groups(images):
    """Get the prediction groups
    Args:
    images (list): A list of images
    return: A list of prediction groups
    """
    return pipeline.recognize(images)

def create_predictions_dict(predictions):
    """Create a dictionary of predictions
    Args:
    predictions (list): A list of predictions
    return: A dictionary of predictions
    """
    predictions_dict = {}
    for i, prediction in enumerate(predictions):
        text = prediction[0]
        x1, y1 = int(round(prediction[1][0][0])), int(round(prediction[1][0][1]))
        x2, y2 = int(round(prediction[1][2][0])), int(round(prediction[1][2][1]))
        predictions_dict[i] = {
            "text": text,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
    return predictions_dict


def cv2_annotations(predictions, image, with_text=False):
    """Draw the annotations on the image
    Args:
    predictions (list): A list of predictions
    image (numpy array): The image to draw the annotations on
    with_text (bool): Whether to draw the text on the image
    return: The image with the annotations
    """
    predictions_dict = {}
    for i, prediction in enumerate(predictions):
        text = prediction[0]
        x1, y1 = int(round(prediction[1][0][0])), int(round(prediction[1][0][1]))
        x2, y2 = int(round(prediction[1][2][0])), int(round(prediction[1][2][1]))
        cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
        if with_text:
            cv2.putText(image, text, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def save_image(image, output_name):
    """Save the image
    Args:
    image (numpy array): The image to save
    output_name (str): The name of the output file
    """
    cv2.imwrite(output_name, image)


def process_single_file(file):
    """Run the pipeline on a single file
    Args:
    file (str): The file to run the pipeline on
    return: A dictionary of predictions and the image with the annotations
    """

    image = keras_ocr.tools.read(file)
    prediction_groups = pipeline.recognize([image])
    predictions = prediction_groups[0]
    predictions_dict = create_predictions_dict(predictions)
    image2 = cv2_annotations(predictions, cv2.imread(file))
    dir, filename = os.path.split(file)
    fname, ext = os.path.splitext(filename)
    outfile = fname + '_Keras.png'
    outfile = os.path.join('Results', outfile)
    cv2.imwrite(outfile, image2)
    return predictions_dict, image2

