import os, sys
import time
import argparse
import json, yaml
import boto3
import pyproj
import numpy as np

from dotenv import load_dotenv
from botocore.exceptions import ClientError
from logzero import logger
from tqdm import tqdm
from osgeo import gdal


class TIF2COG:
    
    def __init__(self, s3_access_key, s3_secret_key, s3_endpoint, s3_virtual_hosting, workdir):
        
        session = boto3.session.Session()
        
        self.s3_access_key = s3_access_key
        self.s3_secret_key = s3_secret_key
        self.s3_endpoint = s3_endpoint
        self.s3_virtual_hosting = s3_virtual_hosting
        self.workdir = workdir

        self.client = session.client(
            service_name = 's3',
            aws_access_key_id = self.s3_access_key,
            aws_secret_access_key = self.s3_secret_key,
            endpoint_url = f"https://{self.s3_endpoint}",
        )
        
        gdal.SetConfigOption('AWS_ACCESS_KEY_ID', self.s3_access_key)
        gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', self.s3_secret_key)
        gdal.SetConfigOption('AWS_S3_ENDPOINT', self.s3_endpoint)
        gdal.SetConfigOption('AWS_VIRTUAL_HOSTING', self.s3_virtual_hosting)
        
        return
    
    
    def get_workload(self, s3_bucket, s3_prefix):
        
        paginator = self.client.get_paginator('list_objects_v2')
        
        page_iterator = paginator.paginate(
            Bucket=s3_bucket,
            Prefix=s3_prefix,
            PaginationConfig={
                'MaxItems': 100000,
                'PageSize': 100,
            }
        )
    
        workload = [] # images to process - one S3 key per image
    
        for idx, page in enumerate(page_iterator):
            workload += [x['Key'] for x in page['Contents'] if x['Key'].endswith('.tif')]
            
        return workload
    
    
    def reproject_and_gen_overviews(self, s3_key, s3_bucket, s3_bucket_tif):
        
        basename = os.path.basename(s3_key)
        src_objname = f"/vsis3/{s3_bucket}/{s3_key}"
        tmp_file = os.path.join(self.workdir, basename)
        dst_objname = f"{s3_bucket_tif}/{basename}"
        
        try:
            self.client.head_object(Bucket=s3_bucket, Key=dst_objname)
            logger.info("Destination object already exists => skipping")
            
            return False
        
        except ClientError:
            # Not found
            
            t0 = time.time()
            src_ds = gdal.Open(src_objname, gdal.GA_ReadOnly)
            
            gdalWarpOptions = gdal.WarpOptions(format='GTiff',
                                               srcSRS='EPSG:2056',
                                               dstSRS='EPSG:3857',
                                               #resampleAlg='bilinear',
                                               resampleAlg='near',
                                               srcNodata=0,
                                               dstNodata=0,
                                               multithread=False
            )
            
            dst_ds = gdal.Warp(tmp_file,
                      src_ds,
                      options=gdalWarpOptions
            )
            
            
            t1 = time.time()
            logger.debug(f"Time taken to reproject: {t1-t0:.2f} s")
            
            # gen overviews
            dst_ds.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32, 64, 128, 256])
            t2 = time.time()
            logger.debug(f"Time taken to generate overviews: {t2-t1:.2f} s")
            
            # fix color interpretation
#             for i in range(1, dst_ds.RasterCount+1):
#                 band = dst_ds.GetRasterBand(i)
#                 if i == 1: # NIR
#                     band.SetColorInterpretation(0) # 0 = undefined, cf. https://github.com/rasterio/rasterio/issues/100
#                 else: # RGB
#                     band.SetColorInterpretation(i + 1) # cf. https://gis.stackexchange.com/a/414699
#                 del band
                
            # Once we're done, close properly the dataset
            dst_ds = None
            src_ds = None
            
            # copy to S3
            try:
                self.client.upload_file(tmp_file, s3_bucket, dst_objname)
            except ClientError as e:
                logger.error(e)
                
            t3 = time.time()           
            logger.debug(f"Time taken to upload to S3: {t3-t2:.2f} s")
                
            # delete temporary file
            if os.path.isfile(tmp_file):
                os.remove(tmp_file)
                
            return True
    
    
    def generate_tiff_with_overviews(self, s3_key, s3_bucket, s3_prefix_tif):
        
        basename = os.path.basename(s3_key)
        src_objname = f"/vsis3/{s3_bucket}/{s3_key}"
        tmp_file = os.path.join(self.workdir, basename)
        dst_objname = f"{s3_prefix_tif}/{basename}"
        
        try:
            self.client.head_object(Bucket=s3_bucket, Key=dst_objname)
            logger.info("Destination object already exists => skipping")
            
            return False
        
        except ClientError:
            # Not found
            
            src_ds = gdal.Open(src_objname, gdal.GA_ReadOnly)

            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.CreateCopy(tmp_file, src_ds, strict=0)

            dst_ds.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32, 64, 128, 256])

            # the following two lines are VERY OPINIONATED
            epsg2056 = pyproj.CRS.from_epsg(2056)
            dst_ds.SetProjection(epsg2056.to_wkt())

            # Once we're done, close properly the dataset
            dst_ds = None
            src_ds = None

            # copy to S3
            try:
                self.client.upload_file(tmp_file, s3_bucket, dst_objname)
            except ClientError as e:
                logger.error(e)
                
            # delete temporary file
            if os.path.isfile(tmp_file):
                os.remove(tmp_file)
                
            return True
        
        
    def compute_stats(self, s3_key, s3_bucket, s3_prefix_tif):
        
        basename = os.path.basename(s3_key)
        src_objname = f"/vsis3/{s3_bucket}/{s3_prefix_tif}/{basename}"
        
        src_ds = gdal.Open(src_objname, gdal.GA_ReadOnly)
        stats = {}
        for i in range(1, src_ds.RasterCount+1):
            # cf. https://arijolma.org/Geo-GDAL/2.4/classGeo_1_1GDAL_1_1Band.html#a8e2361c3a2d76bbfa7253370c78990cf
            _min, _max, _mean, _stddev = src_ds.GetRasterBand(i).GetStatistics(1, 1)
            stats[str(i)] = {}
            stats[str(i)]['min'] = _min
            stats[str(i)]['max'] = _max
            stats[str(i)]['mean'] = _mean
            stats[str(i)]['stddev'] = _stddev
            
        return stats
    
    
    @staticmethod
    def summarize_stats(stats, r_idx, g_idx, b_idx, nir_idx):
        
        summary = {}
        
        rgb_stats = {
            k: {
                r_idx: v[str(r_idx)], 
                g_idx: v[str(g_idx)], 
                b_idx: v[str(b_idx)]
            } for k, v in stats.items()
        }
        
        nir_stats = {
            k: {
                nir_idx: v[str(nir_idx)]
            } for k, v in stats.items()
        }

        rgb_stats_list = [list(x.values()) for x in rgb_stats.values()]
        rgb_flat_stats_list = [e for v in rgb_stats_list for e in v]

        nir_stats_list = [list(x.values()) for x in nir_stats.values()]
        nir_flat_stats_list = [e for v in nir_stats_list for e in v]

        fact = 2.0
        
        rgb_mins = [x['mean']-fact*x['stddev'] for x in rgb_flat_stats_list]
        rgb_maxs = [x['mean']+fact*x['stddev'] for x in rgb_flat_stats_list]

        nir_mins = [x['mean']-fact*x['stddev'] for x in nir_flat_stats_list]
        nir_maxs = [x['mean']+fact*x['stddev'] for x in nir_flat_stats_list]

        summary['rgb_min'] = max(np.mean(rgb_mins)-np.std(rgb_mins), 0)
        summary['rgb_max'] = min(np.mean(rgb_maxs)+np.std(rgb_maxs), 65535)

        summary['nir_min'] = max(np.mean(nir_mins)-np.std(nir_mins), 0)
        summary['nir_max'] = min(np.mean(nir_maxs)+np.std(nir_maxs), 65535)

        return summary
        
        
    def generate_cogs(self, s3_key, s3_bucket, s3_prefix_tif, s3_prefix_cog, summary_stats):
        
        basename = os.path.basename(s3_key)
        src_objname = f"/vsis3/{s3_bucket}/{s3_prefix_tif}/{basename}"
        
        tmp_file = os.path.join(self.workdir, basename)
        dst_objname = f"{s3_prefix_cog}/{basename}"
        
        try:
            self.client.head_object(Bucket=s3_bucket, Key=dst_objname)
            logger.info("Destination object already exists => skipping")
            return False
        
        except ClientError:
        
            t0 = time.time()
            src_ds = gdal.Open(src_objname, gdal.GA_ReadOnly)
            
            # doc: https://gdal.org/api/python/osgeo.gdal.html?highlight=translate#osgeo.gdal.Translate
            dst_ds = gdal.Translate(
                tmp_file, 
                src_ds, 
                outputType=gdal.GDT_Byte, 
                creationOptions=["TILED=YES", "COPY_SRC_OVERVIEWS=YES", "COMPRESS=NONE"],
                scaleParams=[
                    [summary_stats['nir_min'], summary_stats['nir_max'], 0, 255], 
                    [summary_stats['rgb_min'], summary_stats['rgb_max'], 0, 255], 
                    [summary_stats['rgb_min'], summary_stats['rgb_max'], 0, 255], 
                    [summary_stats['rgb_min'], summary_stats['rgb_max'], 0, 255]]
            )
        
            # Once we're done, close properly the dataset
            dst_ds = None
            src_ds = None
            
            t1 = time.time()   
            logger.debug(f"Time taken by GDAL Translate: {t1-t0:.2f} s")

            # copy to S3
            try:
                self.client.upload_file(tmp_file, s3_bucket, dst_objname)

            except ClientError as e:
                logger.error(e)
                
            t2 = time.time()           
            logger.debug(f"Time taken to upload to S3: {t2-t1:.2f} s")

            # delete temporary file
            if os.path.isfile(tmp_file):
                os.remove(tmp_file)
                
            return True
        
    
if __name__ == "__main__":
    
    tic = time.time()
    
    parser = argparse.ArgumentParser(description="This quite opinionated script converts GeoTIFFs to COGs.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader) #[os.path.basename(__file__)]
    
    ### --- read config --- ###
    load_dotenv() # take environment variables from .env
    
    S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
    S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
    S3_ENDPOINT = os.getenv('S3_ENDPOINT')
    S3_VIRTUAL_HOSTING = os.getenv('S3_VIRTUAL_HOSTING')
    S3_BUCKET = os.getenv('S3_BUCKET')
    
    S3_PREFIX_IN = cfg['S3_PREFIX_IN']
    S3_PREFIX_TIF = cfg['S3_PREFIX_TIF']
    S3_PREFIX_COG = cfg['S3_PREFIX_COG']
    
    WORKDIR = cfg['WORKDIR']
    
    NIR_BAND_NO = cfg['NIR_BAND_NO']
    R_BAND_NO = cfg['R_BAND_NO']
    G_BAND_NO = cfg['G_BAND_NO']
    B_BAND_NO = cfg['B_BAND_NO']
    
    DO_STEP1 = cfg['DO_STEP1']
    DO_STEP2 = cfg['DO_STEP2']
    DO_STEP3 = cfg['DO_STEP3']
    ### --- ###
    
    
    tif2cog = TIF2COG(S3_ACCESS_KEY, S3_SECRET_KEY, S3_ENDPOINT, S3_VIRTUAL_HOSTING, WORKDIR)
    
    workload = tif2cog.get_workload(S3_BUCKET, S3_PREFIX_IN)
    
    # debugging hack
    # workload = workload[0:4]
    
    logger.info(f"Number of images to process: {len(workload)}")
    
    if DO_STEP1:
        logger.info("Step #1: Reproject (EPSG:2056 -> EPSG:3857) and generate overviews...")
        t1 = time.time()
        cnt = 0
        
        for s3_key in tqdm(workload):

            ok = tif2cog.reproject_and_gen_overviews(s3_key, S3_BUCKET, S3_PREFIX_TIF)
            if ok:
                cnt += 1

        t2 = time.time()
        logger.info(f"...done. {cnt} image{'s' if cnt != 1 else ''} were processed ({cnt/(t2-t1):.2f} images/s)")
        
    
#     if DO_STEP1:
#         logger.info("Step #1: Generate TIFFs with overviews...")
#         t1 = time.time()
#         cnt = 0
        
#         for s3_key in tqdm(workload):

#             ok = tif2cog.generate_tiff_with_overviews(s3_key, S3_BUCKET, S3_PREFIX_TIF)
#             if ok:
#                 cnt += 1
                
#         t2 = time.time()

#         logger.info(f"...done. {cnt} image{'s' if cnt != 1 else ''} were processed ({cnt/(t2-t1):.2f} images/s)")

    if DO_STEP2:
        
        CACHE_FILE = 'output/stats.json'
        
        logger.info("Step #2: Computing statistics...")
        t1 = time.time()
        cnt = 0
        
        if os.path.isfile(CACHE_FILE) and False: # <== /!\
            with open(CACHE_FILE, 'r') as fp:
                stats = json.load(fp)
        else:
            stats = {}
            for s3_key in tqdm(workload):

                # if s3_key != "02_Data/initial/images_RS/20180626_1129_12504_0_30.tif":
                #     continue

                _stats = tif2cog.compute_stats(s3_key, S3_BUCKET, S3_PREFIX_TIF)
                stats[s3_key] = _stats
                cnt += 1

            with open(CACHE_FILE, 'w') as fp:
                json.dump(stats, fp)

        # analyze stats
        summary = tif2cog.summarize_stats(stats, r_idx=R_BAND_NO, g_idx=G_BAND_NO, b_idx=B_BAND_NO, nir_idx=NIR_BAND_NO)

        t2 = time.time()
        logger.info(f"...done. {cnt} image{'s' if cnt != 1 else ''} were processed ({cnt/(t2-t1):.2f} images/s)")
        logger.info(f"Summary statistics: {json.dumps(summary, indent=4)}")
    
    if DO_STEP3:
        
        logger.info("Step #3: Generate COGs...")
        t1 = time.time()
        cnt = 0

        for s3_key in tqdm(workload):
            
            # if s3_key != "02_Data/initial/images_RS/20180626_1129_12504_0_30.tif":
            #     continue
            
            ok = tif2cog.generate_cogs(s3_key, S3_BUCKET, S3_PREFIX_TIF, S3_PREFIX_COG, summary)
            if ok:
                cnt += 1

        t2 = time.time()
        logger.info(f"...done. {cnt} image{'s' if cnt != 1 else ''} were processed ({cnt/(t2-t1):.2f} images/s)")
    
        
    toc = time.time()
    logger.info(f"Total elapsed time = {toc-tic:.2f} s")