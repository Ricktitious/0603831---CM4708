import boto3
import os
import cv2
import json

# Apologies, I can't share my AWS credentials
# if you have an AWS account, you can set the environment variables below
# however, I've already saved the response from textract to a json file
# which will load automatically if detected

os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_Key'] = ''

# AWS Access Key ID and Secret Access Key
# These are stored in the environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
REGION_NAME = 'eu-west-1'

# Create a Textract client
textract = boto3.client('textract', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                        region_name=REGION_NAME)


def detect_document_text(file):
    """Detects text in the document
    Args:
    file (str): A filepath
    return: A list of blocks"""
    # Call Amazon Textract
    file_as_bytes = open(file, 'rb').read()
    response = textract.detect_document_text(Document={'Bytes': file_as_bytes})

    # hand file name
    path, name = os.path.split(file)
    f, ext = os.path.splitext(name)
    save = os.path.join(path, f + '.json')
    # save the response
    with open(save, 'w') as f:
        json.dump(response, f)

    # get the text blocks
    blocks = response['Blocks']
    return blocks

def analyse_document(file, feature_types=['TABLES']):
    """Detect text, tables, forms, and key-value pairs in a document
    Args:
    file (str): A filepath
    feature_types (list): A list of feature types
    return: A list of blocks
        """
    # open the file
    file_as_bytes = open(file, 'rb').read()
    # Call Amazon Textract with analyse_document
    response = textract.analyze_document(Document={'Bytes': file_as_bytes}, FeatureTypes=feature_types)
    return response

def get_text_from_block(block):
    """Get text from a block
    Args:
    block (dict): A block
    return: A string"""
    text = ''
    if block['BlockType'] == 'WORD':
        text = block['Text']
        return text

def get_cv2_boundingbox_from_block(block, height, width):
    """Get bounding box from a block
    Args:
    block (dict): A block
    height (int): The height of the image
    width (int): The width of the image
    return: A tuple of two tuples"""
    if block['BlockType'] == 'WORD':
        boundingbox = block['Geometry']['BoundingBox']
        x1 = int(boundingbox['Left'] * width)
        y1 = int(boundingbox['Top'] * height)
        x2 = int((boundingbox['Left'] + boundingbox['Width']) * width)
        y2 = int((boundingbox['Top'] + boundingbox['Height']) * height)
        pt1, pt2 = (x1, y1), (x2, y2)
        return pt1, pt2

def process_blocks_to_boundingbox_dict(blocks, height, width):
    """Process blocks to a dictionary of bounding boxes
    Args:
    blocks (list): A list of blocks
    height (int): The height of the image
    width (int): The width of the image
    return: A dictionary of bounding boxes"""
    boundingbox_dict = {}
    for i, block in enumerate(blocks):
        text = get_text_from_block(block)
        if text is not None:
            pt1, pt2 = get_cv2_boundingbox_from_block(block, height, width)
            boundingbox_dict[i] = {'text': text, 'boundingbox': {'pt1': pt1, 'pt2': pt2}}
    return boundingbox_dict

def annotate_image_with_boundingboxes(image, boundingbox_dict, with_text=False, color=(0, 0, 255), thickness=2):
    """Annotate an image with bounding boxes
    Args:
    image (numpy.ndarray): An image
    boundingbox_dict (dict): A dictionary of bounding boxes
    with_text (bool): Whether to write the text on the image
    color (tuple): The color of the bounding box
    thickness (int): The thickness of the bounding box
    return: An annotated image"""
    for key, value in boundingbox_dict.items():
        text = value['text']
        pt1 = value['boundingbox']['pt1']
        pt2 = value['boundingbox']['pt2']
        cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=color, thickness=thickness)
        if with_text:
            cv2.putText(image, text, pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def load_response_json(file):
    """Load a json file
    Args:
    file (str): A filepath
    return: textract json response"""
    with open(file, 'r') as f:
        response = json.load(f)
    return response

def process_single_file(file):
    """Process a single file
    Args:
    file (str): An image filepath
    return: A dictionary of bounding boxes and an annotated image
    """

    image = cv2.imread(file)
    height, width, channels = image.shape

    # if textract data already exists, load it
    dir, filename = os.path.split(file)
    fname, ext = os.path.splitext(filename)
    jsonfile = fname + '.json'
    jsonfile = os.path.join(dir, jsonfile)
    if os.path.exists(jsonfile):
        response = load_response_json(jsonfile)
        blocks = response['Blocks']
    # else, call textract
    else:
        blocks = detect_document_text(file)
    # process the blocks
    boundingbox_dict = process_blocks_to_boundingbox_dict(blocks, height, width)
    # annotate the image
    image2 = annotate_image_with_boundingboxes(image, boundingbox_dict)
    # save the image
    outfile = fname + '_textract.png'
    outfile = os.path.join('Results', outfile)
    cv2.imwrite(outfile, image2)
    return boundingbox_dict, image2


