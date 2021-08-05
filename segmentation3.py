# module load  Python/3.7.4-GCCcore-8.3.0
# conda activate seg

from cytomine import Cytomine
from cytomine import CytomineJob
from cytomine.models import Property, Annotation, AnnotationTerm, AnnotationCollection

import logging
import sys

import json as js

import os
from pathlib import Path

#from params import args

#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from keras.preprocessing.image import img_to_array, load_img
from keras.applications.imagenet_utils import preprocess_input
from models.model_factory import make_model
from os import path, mkdir, listdir
import numpy as np
from tensorflow import keras # th

np.random.seed(1)
import random

random.seed(1)
import tensorflow as tf

#tf.set_random_seed(1)
tf.random.set_seed(1) # TH, for recent TF
import timeit
import cv2
import imutils
from tqdm import tqdm

import sys
from argparse import ArgumentParser

from shapely.geometry import Point

#test_folder = args.test_folder
#test_pred = os.path.join(args.out_root_dir, args.out_masks_folder)

all_ids = []
all_images = []
all_masks = []

#OUT_CHANNELS = args.out_channels

# Crappy Python does not support overloading....

# Download file from Cytomine
#def CytomineDownloadFile(host,public_key,private_key,query,output_file):
#        with Cytomine(host=host, public_key=public_key, private_key=private_key) as cytomine:
#                status = cytomine.get_instance().download_file(url=query,destination=output_file)
#                return(status)
def CytomineDownloadFile(CytoJob,query,output_file):
    status = CytoJob.get_instance().download_file(url=query,destination=output_file)
    return(status)

# Generic json get
#def CytomineGet(host,public_key,private_key,query):
#        with Cytomine(host=host, public_key=public_key, private_key=private_key) as cytomine:
#                json_return = cytomine.get_instance().get(query)
#                return(json_return)
def CytomineGet(CytoJob,query):
    json_return = CytoJob.get_instance().get(query)
    return(json_return)

# Generic json post. Generally used for posting new annotations etc.
#def CytominePost(host,public_key,private_key,query,data):
#        with Cytomine(host=host, public_key=public_key, private_key=private_key) as cytomine:
#                json_return = cytomine.get_instance().post(query,data)
#                return(json_return)
def CytominePost(CytoJob,query,data):
    json_return = CytoJob.get_instance().post(query,data)
    return(json_return)

# Generate point annotation json
def GenPointJson(parent, point_coordinate, shift_coordinate,termID,user_id):
    #term_ID = "214247"
    #user_ID = "58"
    x = point_coordinate[0]+shift_coordinate[0]
    y = point_coordinate[1]+shift_coordinate[1]
    #json = '{"location":["POINT ( '+str(x)+' '+str(y)+' )"],"image":"'+str(parent["image"])+'","project":"'+str(parent["project"])+'","user":"'+str(user_id)+'","term":["'+str(termID)+'"]}'  # OK for userannotation.json
    #json = '{"location":["POINT ( '+str(x)+' '+str(y)+' )"],"image":"'+str(parent["image"])+'","project":"'+str(parent["project"])+'","idTerm":["'+str(termID)+'"]}' #somewhat ok
    #json = '{"location":["POINT ( '+str(x)+' '+str(y)+' )"],"image":"'+str(parent["image"])+'","project":"'+str(parent["project"])+'","idExpectedTerm":["'+str(termID)+'"]}'
    #json = '{"location":["POINT ( '+str(x)+' '+str(y)+' )"],"image":"'+str(parent["image"])+'","project":"'+str(parent["project"])+'","linkedAnnotations":["'+str(471807)+'"]}'
    #json = '{"location":["POINT ( '+str(x)+' '+str(y)+' )"],"image":"'+str(parent["image"])+'","project":"'+str(parent["project"])+'","annotationLinks":"'+str(472970)+ '","linkedAnnotations":["'+str(471807)+'"]}'
    #json = '{"location":["POINT ( '+str(x)+' '+str(y)+' )"],"image":"'+str(parent["image"])+'","project":"'+str(parent["project"])+'","idTerm":'+str(214247)+'}' # 404
    #json = '{"location":["POINT ( '+str(x)+' '+str(y)+' )"],"image":"'+str(parent["image"])+'","project":"'+str(parent["project"])+'","idTerm":"'+str(214247)+'"}' # 404
    #json = '{"location":["POINT ( '+str(x)+' '+str(y)+' )"],"image":"'+str(parent["image"])+'","project":"'+str(parent["project"])+'","term":"'+str(214247)+'"}' # 404 boolobject...
    #json = '{"location":["POINT ( '+str(x)+' '+str(y)+' )"],"image":"'+str(parent["image"])+'","project":"'+str(parent["project"])+'","term":["'+str(214247)+'"]}' # 404 boolobject...
    #json = '{"location":["POINT ( '+str(x)+' '+str(y)+' )"],"image":"'+str(parent["image"])+'","project":"'+str(parent["project"])+'","term":['+str(214247)+']}' # OK!
    json = '{"location":["POINT ( '+str(x)+' '+str(y)+' )"],"image":"'+str(parent["image"])+'","project":"'+str(parent["project"])+'","term":['+str(termID)+']}' # OK!

    #json = '{"location":["POINT ( '+str(x)+' '+str(y)+' )"],"image":"'+str(parent["image"])+'","project":"'+str(parent["project"])+'","user":"'+str(user_id)+'"}'
    print("Point returning: ",json)
    return(json)

def GenPropertyJson(annotation_id, key , value):
    json = '{"domainIdent":"'+str(annotation_id)+',"domainClassName="be.cytomine.ontology.UserAnnotation","key":"'+key+'","value":"'+value+'"}'
    return(json)

def preprocess_inputs(x):
    #return preprocess_input(x, mode=args.preprocessing_function)
    return preprocess_input(x, mode=params.preprocessing_function)


def segmentation(logger,modelfiles,basedir=".",models_dir = "nn_models",predictions_dir="predictions",images_dir="images",out_channels=3,writedebug=False):
    t0 = timeit.default_timer()
    
    predictions_dir = basedir + "/" + predictions_dir
    images_dir = basedir + "/" + images_dir
    models_dir = basedir + "/" + models_dir
    out_channels = int(out_channels)
    
    print("Base directory",basedir)
    print("Predictions directory:",predictions_dir)
    print("Images directory:",images_dir)
    print("Models directory:",models_dir)
    print("Channels:",out_channels)

    #weights = [os.path.join(args.models_dir, m) for m in args.models]
    #weights = [os.path.join(params.models_dir, m) for m in params.models]
    weights = [os.path.join(models_dir, m) for m in modelfiles]
    models = []
    print("Loading models")
    for w in weights:
        #modelfile= Path("built_models/{}.hdf5".format(w))
        #w = models_dir+"/"+w+".hdf5"
        #w = w+".hdf5"
        modelfile = Path(w)

        #modelfile= Path("models/{}.hdf5".format(w))
        if modelfile.is_file():
            logger.update("Loading model "+str(modelfile),1,1)
            print(" Loading existing model",str(modelfile))
            model = keras.models.load_model(modelfile)
            models.append(model)
        else:
            print(" Model file",modelfile,"not found ")
            #print(" Building model {} from weights {} ".format(params.network, w))
            ##model = make_model(args.network, (None, None, 3))
            #model = make_model(params.network, (None, None, 3))
            ##print("Building model {} from weights {} ".format(args.network, w))
            #model.load_weights(w)
            #model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
            #model.save(filepath="built_models/{}.hdf5".format(w))
        #        models.append(model)


    #os.makedirs(test_pred, exist_ok=True)
    #os.makedirs(params.predictions_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    print('Predicting nuclei')
    #for d in tqdm(listdir(test_folder)):
    #for d in tqdm(listdir(params.images_dir)):
    retval = {} # filename + coordinates
    for d in tqdm(listdir(images_dir)):
        logger.update("Predicting nuclei from image"+d,1,1)
        final_mask = None
        for scale in range(1):
            #dd = params.predictions_dir + "/" + d
            dd = predictions_dir + "/" + d
            os.makedirs(dd, exist_ok=True)
            #f = params.images_dir + "/" + d
            f = images_dir + "/" + d
            print("Processing image {}".format(f))
            img = cv2.imread(f, cv2.IMREAD_COLOR)[...,::-1]
            img_orig = cv2.imread(f, cv2.IMREAD_UNCHANGED)
   
            #Handle alpha channel mask
            if img_orig.ndim == 2:
                num_channels=1
            else:
                num_channels=img_orig.shape[-1]

            print("Image channels",num_channels)
            if num_channels==4:
                print("Applying alpha channel")
                img_alpha = img_orig[:,:,3]
                #if params.writedebug:
                if writedebug:
            #dd = params.predictions_dir + "/" + d
                    dd = predictions_dir + "/" + d
                    cv2.imwrite(dd+"/alpha-"+d, img_alpha, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                img[img_alpha == 0] = (255,255,255)

            if final_mask is None:
                #final_mask = np.zeros((img.shape[0], img.shape[1], int(params.out_channels)))
                final_mask = np.zeros((img.shape[0], img.shape[1], int(out_channels)))
            if scale == 1:
                img = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            elif scale == 2:
                img = cv2.resize(img, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)

            x0 = 16
            y0 = 16
            x1 = 16
            y1 = 16
            if (img.shape[1] % 32) != 0:
                x0 = int((32 - img.shape[1] % 32) / 2)
                x1 = (32 - img.shape[1] % 32) - x0
                x0 += 16
                x1 += 16
            if (img.shape[0] % 32) != 0:
                y0 = int((32 - img.shape[0] % 32) / 2)
                y1 = (32 - img.shape[0] % 32) - y0
                y0 += 16
                y1 += 16
            img0 = np.pad(img, ((y0, y1), (x0, x1), (0, 0)), 'symmetric')

            # inp0 = []
            # inp1 = []
            # for flip in range(2):
            #     for rot in range(4):
            #         if flip > 0:
            #             img = img0[::-1, ...]
            #         else:
            #             img = img0
            #         if rot % 2 == 0:
            #             inp0.append(np.rot90(img, k=rot))
            #         else:
            #             inp1.append(np.rot90(img, k=rot))
            #
            # inp0 = np.asarray(inp0)
            # inp0 = preprocess_inputs(np.array(inp0, "float32"))
            # inp1 = np.asarray(inp1)
            # inp1 = preprocess_inputs(np.array(inp1, "float32"))

            # mask = np.zeros((img0.shape[0], img0.shape[1], OUT_CHANNELS))

            # for model in models:
            #     pred0 = model.predict(inp0, batch_size=1)
            #     pred1 = model.predict(inp1, batch_size=1)
            #     j = -1
            #     for flip in range(2):
            #         for rot in range(4):
            #             j += 1
            #             if rot % 2 == 0:
            #                 pr = np.rot90(pred0[int(j / 2)], k=(4 - rot))
            #             else:
            #                 pr = np.rot90(pred1[int(j / 2)], k=(4 - rot))
            #             if flip > 0:
            #                 pr = pr[::-1, ...]
            #             mask += pr  # [..., :2]

            #mask = np.zeros((img0.shape[0], img0.shape[1], OUT_CHANNELS))
            #mask = np.zeros((img0.shape[0], img0.shape[1], int(params.out_channels)))
            mask = np.zeros((img0.shape[0], img0.shape[1], int(out_channels)))
            for model in models:
                inp = preprocess_inputs(np.array([img0], "float32"))
                pred = model.predict(inp)
                mask += pred[0]

            mask /= (len(models))
            mask = mask[y0:mask.shape[0] - y1, x0:mask.shape[1] - x1, ...]
            if scale > 0:
                mask = cv2.resize(mask, (final_mask.shape[1], final_mask.shape[0]))
            final_mask += mask
        final_mask /= 1
        #if OUT_CHANNELS == 2:
        #if params.out_channels == 2:
        if out_channels == 2:
            final_mask = np.concatenate([final_mask, np.zeros_like(final_mask)[..., 0:1]], axis=-1)
        final_mask = final_mask * 255
        final_mask = final_mask.astype('uint8')

	# drop green
        nogreen_img = final_mask.copy()
        nogreen_img[:,:,1] = np.zeros([nogreen_img.shape[0],nogreen_img.shape[1]])
        nogreen_img_grayscale = cv2.cvtColor(nogreen_img,cv2.COLOR_RGB2GRAY)

        final_mask = nogreen_img

	# Contour based nuclei counting
        grayscale_orig = cv2.cvtColor(nogreen_img,cv2.COLOR_RGB2GRAY)
	# Pad by 2 pixels to include edge contours
        grayscale = np.pad(grayscale_orig.copy(),pad_width=2,mode='constant',constant_values=0)
        thresh = cv2.threshold(grayscale, 0 , 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros(thresh.shape)
        contour_img = cv2.drawContours(final_mask, contours[0], -1, (0,255,75), 2)

        cv2.imwrite(dd+"/contours-"+d, contour_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        cnts = imutils.grab_contours(contours)
        #overlayimg = img.clone()
        overlayimg = img.copy()
        nuclei = []
        height, width, channels = img.shape
        print("Image dimensions, height:",height,"width", width, "channels",channels)
        for c in cnts:
             M = cv2.moments(c)
             if M['m00']==0:
                 print("Skipping point as m00==0")
                 cX = 0
                 cY = 0
             else:
                 cX = int(M['m10']/M['m00'])-2 # Take care of padding
                 cY = int(M['m01']/M['m00'])-2
                 cv2.circle(overlayimg,(int(cX),int(cY)),5,(0,0,255),-1)
                 #nuclei.append((cX,cY))
                 nuclei.append((cX,height-cY)) # Y needs to be flipped for Cytomine
             #print("Contour X",cX,"Y",cY)
             #cv2.circle(overlayimg,(int(cX),int(cY)),5,(0,0,255),-1)
             #nuclei.append((cX,cY))
                #print("Centroids " + str(i) + "X:" + str(cX) + "Y:" + str(cY))

        
        retval[d[:-4]]=nuclei
        #retval.update(d=nuclei)
        #cv2.imwrite(dd+"/overlay-"+d, overlayimg, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
        #ret = cv2.connectedComponentsWithStats(grayscale,4,cv2.CV_32S) # Grayscale connected components not very good

        #(numLabels, labels, stats, centroids) = ret
        #print("Number of nuclei:",numLabels)

        #if params.writedebug:
        if writedebug:
            #dd = params.predictions_dir + "/" + d
            dd = predictions_dir + "/" + d
            print("Writing mask images to ",dd)
            os.makedirs(dd, exist_ok=True)

            cv2.imwrite(dd+"/overlay-"+d, overlayimg, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            cv2.imwrite(dd+"/mask_no_green-"+d, nogreen_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            cv2.imwrite(dd+"/mask-"+d, final_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            cv2.imwrite(dd+"/grayscale-"+d, grayscale, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            cv2.imwrite(dd+"/binary-"+d, thresh, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            csvfile = dd+"/"+d+".csv"
            np.savetxt(csvfile,nuclei,"%d",delimiter=",")

        #csvfile = params.predictions_dir+"/"+d+".csv"
        #centroids = centroids.astype(int)
        #np.savetxt(csvfile,centroids,"%d",delimiter=",")

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
    #print(retval)
    return(retval)

if __name__ == '__main__':
    parser = ArgumentParser(prog="segmentation")
    #parser.add_argument('--image', dest='image',
    #                    help="Image file", required=True)
    parser.add_argument('--w',dest='writedebug',help="Write mask images under predictions/<filename>-masks/", action='store_true')
    parser.add_argument('--channels', dest="out_channels",help="Output image channels",default=2)
    parser.add_argument('--basedir', dest="base_dir", help="Base directory",required=True)
    parser.add_argument('--images', dest="images_dir", help="Images directory relative to base directory. Default=images/",required=False,default="images/")
    parser.add_argument('--predictions', dest="predictions_dir", help="Predictions directory relative to base directory. Default=predictions/",required=False,default="predictions/")
    parser.add_argument('--models_dir',dest="models_dir",help="NN models directory relative to base directory. Default=nn_models/",required=False,default="nn_models/")
    #parser.add_argument('--network',dest="network",help="Neural network to use", required=True)
    parser.add_argument('--models',dest="models", help="List of NN models to use", required=True,nargs='+')
    parser.add_argument('--preprocessing',dest="preprocessing_function",help="Preprocessing function", required=True)
    parser.add_argument('--host',dest="server",help="Cytomine server")
    parser.add_argument('--public_key',dest="public_key",help="Cytomine public key")
    parser.add_argument('--private_key',dest="private_key",help="Cytomine private key")
    parser.add_argument('--project_id',dest="project_id",help="Cytomine Project ID")
    parser.add_argument('--software_id',dest="project_id",help="Cytomine Software ID")
    parser.add_argument('--ann_id',dest="ann_id",help="Annotation IDs",nargs='+')
    parser.add_argument('--term_id',dest="term_id",help="Term ID to use",nargs='+')
    parser.add_argument('--add_annotations',dest="add_annotations",help="Add nuclei annotations to Cytomine, default=True", default=False,action='store_true')

    
    params, other = parser.parse_known_args(sys.argv[1:])

    with CytomineJob.from_cli(sys.argv) as cj:
            logger = cj.job_logger()
            #print("Cytomine job")
            #print(cj.parameters())
            #1 Pull images from cytomine
            print("Pulling annotations")
            #print(params.ann_id)
            annotations = {}
            coordshifts = {}
            min_x = 0
            min_y = 0
            num_annotations = len(params.ann_id)
            counter=0;
            for annotation in params.ann_id:
                counter=counter+1
                logger.update("Pulling annotation "+annotation +" image",counter,num_annotations)
                print("Annotation",annotation)
                query="userannotation/"+annotation+"/crop.png?alphaMask=true"
                outputfile="/nucleidetect/"+params.images_dir+str(annotation)+".png"
                print("Query:",query)
                #print("Outputfile:",outputfile)
                #CytomineDownloadFile(params.server,params.public_key,params.private_key,query,outputfile)
                CytomineDownloadFile(cj,query,outputfile)
                query="userannotation/"+annotation+".json"
                #json = CytomineGet(params.server,params.public_key,params.private_key,query)
                json = CytomineGet(cj,query)
                #print(json)
                annotations[annotation] = json
                #print("location:")
                #print(json["location"])
                #print("Annotation coordinates")
                s = json["location"]
                s = s[s.find("(")+1:s.find(")")]
                s = s[s.find("(")+1:s.find(")")]
                #print(s)
                #perimeterpoints = s.split()
                perimeterpoints = s.split(",")
                #print(perimeterpoints)
                #for coords in perimeterpoints:
                #    print(coords.split())
                #perimeterpoints  [(x, y) for x, y in (line.split() for line in perimeterpoints)]
                perimeterpoints = [tuple(coords.split()) for coords in perimeterpoints]
                #print(perimeterpoints)
                min_x = min(perimeterpoints,key = lambda t: t[0])[0]
                min_y = min(perimeterpoints,key = lambda t: t[1])[1]
                max_x = max(perimeterpoints,key = lambda t: t[0])[0]
                max_y = max(perimeterpoints,key = lambda t: t[1])[1]
                print(min_x,min_y,max_x,max_y)
                coordshifts[annotation] = (round(float(min_x)),round(float(min_y))) # for coordinate shift 

                
            #CytomineDownloadFile(params.server,params.public_key,params.private_key,"userannotation/1859/crop.png?draw=true&complete=true&increaseArea=1.25","/nucleidetect/images/test.png")
            #2 Do nuclei detection
            logger.update("Loading Neural Networks and predicting nuclei",1,1)
            coordinates=segmentation(logger,basedir=params.base_dir,modelfiles=params.models,out_channels=params.out_channels,writedebug=params.writedebug)
            print("Main, coordinates:")
            print(coordinates)

            #3 Do coordinate transformation
            print("Annotations")
            print(annotations)
            print("Annotation shift coordinates:")
            print(coordshifts)
            
            #4 Update cyto annotations
            #nucleus_term_ID=214247
            #user_id=58
            print("Using term: ",params.term_id)
            if params.add_annotations:
                logger.update("Adding nuclei annotations to Cytomine",1,1)
                print("Adding nuclei annotations to Cytomine")
                for p in annotations:
                    #print(p)
                    #print(annotations)
                    #print(annotations[p]) # ok
                    #print("Coordinate shifts:",coordshifts[p])
                    #ann_id = p["id"]
                    #print("Annotation id:",ann_id)
                    #def GenPointJson(parent, point_coordinate, shift_coordinates):
                    jsonlist = []
                    #print("jsontesting")
                    for point in coordinates[p]:
                        #print(point)
                        #json = GenPointJson(annotations[p],point,coordshifts[p],nucleus_term_ID,user_id)
                        json = GenPointJson(annotations[p],point,coordshifts[p],params.term_id)
                        jsonlist.append(json)

                    jarr="["+','.join(jsonlist)+"]"
                    #jarr=''.join(jsonlist)
                    #query = "userannotation.json"
                    query = "algoannotation.json"
                    #json = CytominePost(params.server,params.public_key,params.private_key,query,jarr) # Add annotations as single post
                    print("Posting: ",jarr)
                    json = CytominePost(cj,query,jarr) # Add annotations as single post
                    print("Cytomine Response:")
                    print(json["message"])

                    #Skip adding key-value properties for new annotations. Would lead to thousands of updates into cyto -> potential crash/slowdown
                    #addedannotations = json["message"].split(" ")[1].split(",")
                    #for ann in addedannotations:
                    #    print(ann)
                    #    json = GenPropertyJson(ann,"parentAnnotation",p)
                    #    print(json)
                    #    query = "" #placeholder
                    #    json = CytominePost(params.server,params.public_key,params.private_key,query,jarr)

            else:
                #json = CytomineGet(params.server,params.public_key,params.private_key,"userannotation/214205.json")
                #json = '[{"location":["POLYGON (( 1 1 , 500 500 , 500 1 , 1 1 ))"],"image":"1290","project":"180","user":"58","term":["214247"]}]' #ok
                #json = '[{"location":["POINT ( 1000 1000 )"],"image":"1290","project":"180","user":"58","term":["214247"]}]' 
                #print(json)
                #CytominePost(params.server,params.public_key,params.private_key,"userannotation.json",json)
                print("Skipping annotation adding into Cytomine")

            #logger.update("Job completed!",100,100)
            cj.close(value=None)
            #if params.writedebug:

