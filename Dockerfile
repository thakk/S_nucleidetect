FROM tensorflow/tensorflow:2.4.2

RUN	/usr/bin/python3 -m pip install --upgrade pip
RUN	pip install pandas
RUN	pip install matplotlib
RUN	pip install scipy
RUN	pip install scikit-learn
RUN	pip install keras
RUN	pip install keras_applications
RUN	pip install keras_resnet
RUN	pip install opencv-python-headless
RUN	pip install tqdm
RUN	pip install imutils
RUN	curl -s https://packagecloud.io/install/repositories/cytomine-uliege/Cytomine-python-client/script.python.sh | bash
RUN	pip install cytomine-python-client

RUN mkdir /nucleidetect
WORKDIR /nucleidetect
ADD * /nucleidetect/
