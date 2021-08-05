# Adds application to Cytomine
host = "http://cytomine.devel" # Replace to fit your Cytomine deployment
public_key = "79201dc6-1a4f-4111-97a5-d8ac55adcea4" # Replace these with your admin keys
private_key = "ba9b507b-2d70-4605-924b-c6f8b4d55b09"
from cytomine import Cytomine
from cytomine.utilities.descriptor_reader import read_descriptor
with Cytomine(host, public_key, private_key) as c:
	read_descriptor("/nucleidetect/descriptor.json")
