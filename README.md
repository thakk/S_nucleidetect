AI assisted nuclei detection for Cytomine use.

Mostly derived from DSB2018 [ods.ai] topcoders.

# Usage

Running bare container in \"developer mode\" : `singularity exec --containall -B /home/username/git/nucleidetect/:/nucleidetect -B /home/username/git/nucleidetect/pred:/nucleidetect/predictions nucleidetect.sif /bin/bash`



# Configuration

Adding App to Cytomine: `singularity exec --containall -B /home/tomi/git/nucleidetect/:/nucleidetect -B /home/tomi/git/nucleidetect/pred:/nucleidetect/predictions nucleidetect.sif python /nucleidetect/add_app.py`

# Directories

## images

Directory where image files should be placed.

## models

## nn_models

DSB2018 models. Nucleidetect is developed using best_resnet162_2_fold[0-3] models from selim subdir in DSB2018 topcoders.

## predictions


