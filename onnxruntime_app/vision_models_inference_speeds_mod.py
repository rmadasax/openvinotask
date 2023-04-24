#!/usr/bin/python3

import os, sys, getopt, time, json
#from termcolor import cprint
import datetime
import cpuinfo
import platform
import psutil
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
import warnings
import onnxruntime as ort

ort.set_default_logger_severity(3)
warnings.filterwarnings("ignore")

def dumpinfo(log) :
        cpu_info = cpuinfo.get_cpu_info()
        print('#CPU:', cpu_info['brand_raw'], file=log)
        print('#MEM:', psutil.virtual_memory().total, file=log)
        print('#OS:', platform.platform(), file=log)
        #print('#OpenVINO:', openvino.runtime.get_version(), file=log)
    

def title(msg, level=0, bbox=True, end=None,file=None):
	colors = ['red','green', 'blue', 'yellow']
	color = colors[level % len(colors)]
	print(msg,file=file)
    #l=80
	#if bbox : print(' '*(level*8), end=''); cprint('-'*(l-level*8), color)
	#print(' '*(level*8), end=''); cprint(f'| {msg}', color, end=''); print(' '*(20-len(msg)), end=end)
	#if bbox : print(' '*(level*8), end=''); cprint('-'*(l-level*8), color) 


def measure_duration(func):
	def inner(*args, **kwargs):
		start = time.time()
		func(*args, **kwargs)
		elased_time = (time.time() - start)
		return elased_time

	return inner
	
class RandomImageDataset(IterableDataset):
	def __init__(self, amount, img_size):
		super(RandomImageDataset).__init__()
		self.len = amount
		self.img_size = img_size
		self.images = []
		for i in range(self.len):
			image = np.random.randint(0, high=255, size=(800, 800, 3), dtype=int)
			image = self._default_transform()(image)
			image = image[np.newaxis, ...].numpy()

			self.images.append(image)
		
	def _default_transform(self):
		return transforms.Compose([
			transforms.ToTensor(),
			transforms.Resize(self.img_size),
			transforms.ConvertImageDtype(torch.float),
			transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
		])
	
	def __iter__(self):
		for i in range(self.len):
			yield self.images[i]

		
	@staticmethod
	def get_dataloader(img_amount, img_size, batch_size=10, num_workers=8):
		random_dataset = RandomImageDataset(amount=img_amount, img_size=img_size)
		return random_dataset #DataLoader(random_dataset, batch_size=batch_size, num_workers=num_workers)


class OpenVinoTester:
	def __init__(self, dataset, model: str, weight: str, model_name):
		self.dataset = dataset
		self.model = f'torchvision.models.{model}(weights=torchvision.models.{weight}.DEFAULT)'
		model = eval(self.model)
		self.onnx_model_name = self.to_onnx(model, model_name)
		

	def run(self, device, precision="FP32"):
		@measure_duration
		def infer():            
			for batch in self.dataset:
				self.session.run( None, 
								  { self.session.get_inputs()[0].name: batch } 
								);
			   

		if device == "cuda":
			executionProvider = 'CUDAExecutionProvider'
		else:
			executionProvider = 'OpenVINOExecutionProvider'
			#device="CPU" if device=="cpu_openvino" else "GPU"
			device="CPU" 
			device = device + "_" +  precision

		self.session = ort.InferenceSession( self.onnx_model_name, 
											 providers=[executionProvider], 
											 provider_options=[{'device_type' : device},] )

		# Warming up
		for i in range(10):
			infer()

		# Inferencing
		time_taken = infer()
		return time_taken

	def to_onnx(self, model, model_name):
		onnx_model_name = os.path.join("./models", model_name + '.onnx')
		dummy_input = torch.randn(1, 3, 224, 224)
		torch.onnx.export(	model,
							dummy_input,
							onnx_model_name,
							export_params=True,
							opset_version=11,         
							do_constant_folding=True
							)
		return onnx_model_name


if __name__ == '__main__':
	config_file = "./config.json"
	try:
		opts, args = getopt.getopt(sys.argv[1:],"c:",["config="])
	except getopt.GetoptError:
		print("{} -c < json config file >".format(sys.argv[0]))
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("{} -c < json config file >".format(sys.argv[0]))
			sys.exit()
		elif opt in ("-c", "--config"):
			config_file = arg
	
	with open(config_file, 'r') as f:
		config = json.load(f)

	devices = config["devices"]
	precisiona = config["precision"]
	print(precisiona[0]) 
	batch_sizes = config["batch_sizes"]
	models = config["models"]

	if not torch.cuda.is_available():
		devices.remove("cuda")

	columns = ["model", "batch_size", "category"] + devices
	df = pd.DataFrame(columns=columns)
	measured_times = []
	idx = 0;
	now = datetime.datetime.now();
	with open(now.strftime('/data/result_%d%m-%H%M%S.csv'), 'wt') as log:
	    dumpinfo(log)
	    for precision in precisiona:
	        for batch_size in batch_sizes:
		        df = df.append(pd.Series(), ignore_index = True)
		        title(f'batch_size : {batch_size}\t {precision}', level=0,file=log)

		        dataloader = RandomImageDataset.get_dataloader(img_amount=batch_size, img_size=config['input_shape'][2])

		        for model in models:
			        category = model['category']
			        if category:
				        model_name = model['name'].split(".")[-1]
				        df.loc[idx, 'batch_size'] = batch_size
				        df.loc[idx, 'model'] = model_name
				        df.loc[idx, 'category'] = category
				        title(f'{model_name}', level=1,file=log)
				        tester = OpenVinoTester(dataset=dataloader, model=model['name'], weight=model['weights'], model_name=model_name)
				        for device in devices:
					        title(f'{device}', 2, False, '',file=log)

					        try:
						        infer_time = tester.run(device, precision)
						        fps = batch_size/infer_time
						        title(f'{infer_time*1000 : .2f} ms \t {fps : .2f} fps', 0, False,file=log)
						        df.loc[idx, device] = infer_time

					        except Exception as e:
						        print(e)
						        infer_time = 0.0
						        print(f'The Model {model_name} failed on {device}')

				        idx += 1

	    cats = np.unique(df['category'])
