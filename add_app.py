# Adds application to Cytomine. Use Uliege python client
# Still, add github&docker hub to cytomine trusted source
host = "http://cytomine.devel" # Replace to fit your Cytomine deployment
public_key = "02d65f84-c2fa-468a-9c26-53eb4ffdf81e" # Replace these with your admin keys
private_key = "18409efd-693a-4763-aa42-7c4dca85df5a"
from cytomine import Cytomine
from cytomine.utilities.descriptor_reader import read_descriptor
with Cytomine(host, public_key, private_key) as c:
	read_descriptor("/nucleidetect/descriptor.json")
