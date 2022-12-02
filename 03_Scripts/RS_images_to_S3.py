import boto3
from botocore.exceptions import ClientError
from botocore.client import Config
import threading

import glob
from tqdm import tqdm
import os
import sys
import yaml

# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html

# Definitions for functions

def upload_file(file_name, bucket_name, s3_client, object_name=None):
    """Upload a file to an S3 bucket

    - file_name: File to upload
    - bucket: Bucket to upload to
    - object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)
        
    try:
        response = s3_client.upload_file(file_name, bucket_name, object_name)  #, Callback=ProgressPercentage(filepath))
    except ClientError as e:
        print(f"{e}")
        return False
    
    return True

def file_exists_online(bucketname, s3_client, objectname = None):
    """Check if the file already exists in a S3 bucket
    
    - bucketname: bucket to check
    - object_name: S3 object name. If not specified, the file is considered to not exist in the bucket.
    """
    
    if objectname == None:
        return False
    
    try:
        s3_client.head_object(Bucket=bucketname, Key=objectname)
    except ClientError:
        return False
    
    return True


# Definitions for classes

class ProgressPercentage(object):

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()

if __name__=="__main__":

    with open('config.yaml') as fp:
        cfg=yaml.load(fp, Loader=yaml.FullLoader)['RS_images_to_S3.py']

    ACCESS_KEY=cfg['access_key']
    SECRET_KEY=cfg['secret_key']
    ENDPOINT_URL=cfg['url']
    DATAPATH=cfg['datapath']
    BUCKET=cfg['bucket']
    OUTPATH=cfg['outpath']

    dataset=glob.glob(DATAPATH)
    bucket_name=BUCKET

    # Upload the file
    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        endpoint_url=ENDPOINT_URL,
        config=Config(s3={'addressing_style': 'path'})
    )

    successful_upload={'filepath':[],'success':[]}
    for filepath in tqdm(dataset, desc='Uploaded files'):
        
        object_name = OUTPATH + os.path.basename(filepath)
        
        if file_exists_online(bucket_name, s3_client, object_name):
            continue

        successful_upload['filepath'].append(filepath)
        successful_upload['success'].append(upload_file(filepath, bucket_name, s3_client, object_name))

print("Done :)")
print(successful_upload)