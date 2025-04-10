from roboflow import Roboflow
rf = Roboflow(api_key="Gf4VheRtHRk1Af469wxf")
project = rf.workspace("ueli").project("yogurt_detection-j1tob")
version = project.version(1)
dataset = version.download("yolov5")