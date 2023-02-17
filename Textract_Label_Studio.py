import boto3
import os
import cv2
import json
import re
from label_studio_sdk import Client
from botocore import UNSIGNED
from botocore.client import Config

# I'm afraid I can't share my AWS credentials
# unfortunately this is required to load the drawing data from S3

os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_Key'] = ''

# AWS Access Key ID and Secret Access Key
# These are stored in the environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
REGION_NAME = 'eu-west-1'

# bucket for label studio images
bucket = 'label-studio-imgs'

# label studio url and api key - the api key is unique for each user
# you can retireve it when you run label studio locally

LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = ''


# regex patterns
REDUCER_REGEX = r'[0-9]{1,2}\"x[0-9]{1,2}\"'
VALVE_REGEX = r'(PV|PSV|BV|LV|SDSV|SDV|XV|ESV)-?[0-9]{4,5}'
LINE_REGEX = r'[0-9./]{1,5}\"-[a-zA-Z]{1,2}-[0-9]{3,5}'
LINE_REGEX2 = r'[a-zA-Z]{1,2}-[0-9]{3,5}-[0-9a-zA-Z]{3,6}'
LINE_REGEX3 = r'[0-9./]{1,5}\"-?[0-9a-zA-Z]{1,2}-?[0-9]{3,5}-?[a-z-A-Z0-9]{0,4}'
INSTRUMENT_REGEX = r'[a-zA-Z]{0,2}[\/]?[a-zA-Z]{2,3}-?[0-9]{4,5}-?[0-9]{0,2}[\/0-9]{0,2}'
VESSEL_PUMP_REGEX = r'[a-zA-Z]{1}-?[0-9]{4,5}'

def get_s3_client(unsigned=True):
    """Get a boto3 client for S3
    :param unsigned: If True, the client will be unsigned
    :return: boto3 client"""
    if unsigned:
        conf = Config(signature_version=UNSIGNED)
        s3_cli = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                              region_name=REGION_NAME, config=conf)
    else:
        s3_cli = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                              region_name=REGION_NAME)
    return s3_cli

def get_s3_resource():
    """Get a boto3 resource for S3
    :return: boto3 resource"""
    s3_res = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                            region_name=REGION_NAME)
    return s3_res

def save_to_s3(numpy_image, bucket, key):
    """Save a numpy image to s3
    :param numpy_image: numpy image
    :param bucket: bucket name
    :param key: key name
    :return: url of the saved image"""
    print(f'Saving {key} to s3')
    s3_res = get_s3_resource()
    data_serial = cv2.imencode('.png', numpy_image)[1].tobytes()
    s3_res.Object(bucket, key).put(Body=data_serial, ContentType='image/PNG')

    object_acl = s3_res.ObjectAcl(bucket, key)
    response = object_acl.put(ACL='public-read')
    url = get_presigned_url(bucket, key)
    return url

def get_presigned_url(bucket, key, expiration=0):
    """Get a presigned url for an s3 object
    :param bucket: bucket name
    :param key: key name
    :param expiration: expiration in seconds
    :return: url"""
    s3_cli = get_s3_client(unsigned=True)
    url = s3_cli.generate_presigned_url(
        "get_object", ExpiresIn=expiration, Params={"Bucket": bucket, "Key": key}
    )
    return url

def connect_to_label_studio():
    """Connect to the Label Studio API
    :return: Label Studio Client
    """
    # Connect to the Label Studio API and check the connection
    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    ls.check_connection()
    return ls

def create_project(ls, label_config, title='Test Project'):
    """Create a project in Label Studio
    :param ls: Label Studio Client
    :param label_config: Label Studio label config
    :param title: Project title"""
    project = ls.start_project(label_config=label_config, title=title)
    return project

def create_label_config():
    """Create a Label Studio label config
    :return: Label Studio label config"""
    label_config = """
    <View>
      <Image name="image" value="$ocr"/>
      <Labels name="label" toName="image">
        <Label value="Line Number" background="green"/>
        <Label value="Reducer" background="blue"/>
        <Label value="Valve" background="purple"/>
        <Label value="Vessel/Pump" background="red"/>
        <Label value="Instrument" background="yellow"/>
        <Label value="Drawing Number" background="pink"/>
        <Label value="Drawing Rev" background="orange"/>

        <Label value="Text" background="gray"/>
      </Labels>
      <Rectangle name="bbox" toName="image" strokeWidth="3"/>
      <Polygon name="poly" toName="image" strokeWidth="3"/>
      <TextArea name="transcription" toName="image"
                editable="true"
                perRegion="true"
                required="true"
                maxSubmissions="1"
                rows="5"
                placeholder="Recognized Text"
                displayMode="region-list"
                />
    </View>
    """
    return label_config

def get_label_studio_boundingbox_from_block(idx, block, closest_block, height, width):
    """Get a Label Studio bounding box from a Textract block
    :param idx: index of the block
    :param block: Textract block
    :param closest_block: Textract block immediately below the current block
    :param height: height of the image
    :param width: width of the image
    :return: Label Studio results data, closest block
    """
    if block['BlockType'] == 'WORD':
        cls = get_annotation_class(block['Text'])
        if cls is None:
            cls = 'Text'

        text = block['Text']
        boundingbox = block['Geometry']['BoundingBox']
        x = float(boundingbox['Left']) * 100
        y = float(boundingbox['Top']) * 100
        w = float(boundingbox['Width']) * 100
        h = float(boundingbox['Height']) * 100

        combined = False

        if cls == 'Text' and closest_block is not None:
            test_cs = get_annotation_class(closest_block['Text'])
            if test_cs == 'Text':
                # get the annotation immediately below the current annotation
                # and check if combined text is a class
                combined_text = text + "-" + closest_block['Text']
                cls = get_annotation_class(combined_text)
                if cls != 'Text':
                    # if the combined text is a class, then combine the bounding boxes
                    x = min(x, closest_block['Geometry']['BoundingBox']['Left'] * 100)
                    y = min(y, closest_block['Geometry']['BoundingBox']['Top'] * 100)
                    w = max(w, closest_block['Geometry']['BoundingBox']['Width'] * 100)
                    h = h + closest_block['Geometry']['BoundingBox']['Height'] * 100
                    text = combined_text
                    combined = True
                    # we need to flag that we have combined the annotations
                    # so we can remove the annotation from the list
                    # and not draw a bounding box for it

        results = []
        for i in range(3):
            data = {
                "original_width": height,
                "original_height": width,
                "image_rotation": 0,
                "value": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "rotation": 0
                },
                "id": f"bb{str(idx).zfill(5)}",
                "to_name": "image"
            }

            if i == 0:
                data["from_name"] = "bbox"
                data["type"] = "rectangle"
            elif i == 1:
                data["from_name"] = "label"
                data["type"] = "labels"
                data["value"]["labels"] = [cls]
            elif i == 2:
                data["from_name"] = "transcription"
                data["type"] = "textarea"
                data["value"]["text"] = [text]
            results.append(data)
        if combined:
            return results, closest_block
        else:
            return results, None
    else:
        return [], None

def load_response_json(file):
    """Load a Textract response JSON file
    :param file: Textract response JSON file
    :return: Textract response JSON"""
    with open(file, 'r') as f:
        response = json.load(f)
    return response


def filter_blocks_by_regex(blocks, regex):
    """Filter blocks by regex
    :param blocks: Textract blocks
    :param regex: regex to filter blocks
    :return: filtered blocks"""
    filtered_blocks = []
    for block in blocks:
        if block['BlockType'] == 'WORD':
            if re.match(regex, block['Text']):
                filtered_blocks.append(block)
    return filtered_blocks

def find_block_vertically_below(block, blocks):
    """Finds the block that is vertically below the given block.
    :param block: Textract block
    :param blocks: Textract blocks
    :return: Textract block that is vertically below the given block"""

    # get block coordinates
    x0, y0, x1, y1 = block['Geometry']['BoundingBox']['Left'], block['Geometry']['BoundingBox']['Top'], \
                        block['Geometry']['BoundingBox']['Left'] + block['Geometry']['BoundingBox']['Width'], \
                        block['Geometry']['BoundingBox']['Top'] + block['Geometry']['BoundingBox']['Height']
    # get block height
    bbox_height = abs(y1 - y0)
    # get block width
    bbox_width = abs(x1 - x0)
    # get block center
    x_center = x0 + (bbox_width / 2)
    y_center = y0 + (bbox_height / 2)

    # get blocks that are vertically below the given block
    blocks_below = [b for b in blocks if b['Geometry']['BoundingBox']['Top'] > y0 or b['Geometry']['BoundingBox']['Top'] > y1]
    # filter out block that is the same as the given block
    blocks_below = [b for b in blocks_below if b != block]

    # if no blocks are below the given block, return None
    if len(blocks_below) == 0:
        return None

    # limit blocks to those that are within the same column plus a small margin
    a = x0 - bbox_width * 0.3
    b = x1 + bbox_width * 0.3
    refined_blocks_below = []
    for blk in blocks_below:
        bx0, by0, bx1, by1 = blk['Geometry']['BoundingBox']['Left'], blk['Geometry']['BoundingBox']['Top'], \
                        blk['Geometry']['BoundingBox']['Left'] + blk['Geometry']['BoundingBox']['Width'], \
                        blk['Geometry']['BoundingBox']['Top'] + blk['Geometry']['BoundingBox']['Height']
        block_center = bx0 + (abs(bx0 - bx1) / 2)
        if min(a, b) < block_center < max(a, b):
            refined_blocks_below.append(blk)

    # if no blocks are below the given block, return None
    if len(refined_blocks_below) == 0:
        return None

    # get the annotation that is closest to the given annotation
    closest_block = min(refined_blocks_below, key=lambda x: abs(x['Geometry']['BoundingBox']['Top'] - y_center))

    # if the closest annotation is too far away, return None
    closest_block_y_center = closest_block['Geometry']['BoundingBox']['Top'] + (closest_block['Geometry']['BoundingBox']['Height'] / 2)
    if abs(y_center - closest_block_y_center) > bbox_height*1.5:
        return None


    return closest_block


def get_annotation_class(string):
    """Get the annotation class for a given string
    :param string: string to get annotation class for
    :return: annotation class"""
    if re.search(LINE_REGEX, string):
        return 'Line Number'
    # elif re.search(LINE_REGEX2, string):
    #     return 'Line Number'
    elif re.search(LINE_REGEX3, string):
        return 'Line Number'
    elif re.fullmatch(REDUCER_REGEX, string):
        return 'Reducer'
    elif re.fullmatch(VALVE_REGEX, string):
        return 'Valve'
    elif re.fullmatch(INSTRUMENT_REGEX, string):
        return 'Instrument'
    elif re.fullmatch(VESSEL_PUMP_REGEX, string):
        return 'Vessel/Pump'
    else:
        return 'Text'

def process_texract_json_for_label_studio(file, height, width):
    """Process a Textract JSON file for Label Studio
    :param file: Textract JSON file
    :return: Label Studio JSON
    """
    with open(file, 'r') as f:
        json_file = json.load(f)
        blocks = json_file['Blocks']
        blocks = [block for block in blocks if block['BlockType'] == 'WORD']
        results = []
        flagged_blocks = []
        for i, block in enumerate(blocks):
            if block in flagged_blocks:
                continue
            closest_block = find_block_vertically_below(block, blocks)
            res, flagged_block = get_label_studio_boundingbox_from_block(idx=i, block=block, closest_block=closest_block,height=height, width=width)
            if flagged_block is not None:
                flagged_blocks.append(flagged_block)
            results.extend(res)
    return results

def process_file(file):
    # connect to label studio
    ls = connect_to_label_studio()
    # get label config
    label_config = create_label_config()
    # create project
    project = create_project(ls, label_config, title='Example Project')
    # read image
    image = cv2.imread(file)
    img_width, img_height = image.shape[1], image.shape[0]
    # path handling
    path, f = os.path.split(file)
    fname, ext = os.path.splitext(f)
    key = 'Example_Images/' + f
    # save image to s3
    url = save_to_s3(image, bucket, key)
    # load the label config template
    template = json.load(open('ocr_template.json'))
    # update the template with the image url
    template['data']['ocr'] = url
    # process the textract json file
    json_file = file.replace('.png', '.json')
    results = process_texract_json_for_label_studio(json_file, img_height, img_width)
    # update the template with the results
    template['predictions'][0]['result'] = results
    # create task and import it into label studio
    project.import_tasks([template])
    print(f'Created task for {file}')

process_file(r'Data/MAPG-L-0010-040-D-AB00 - 000 - Z17.png')