import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from scipy.misc import imread
from tqdm import tqdm
class GuessWhatDataset(Dataset):
	def __init__(self, img_dir, json_path, dict_path):
		self.guesswhat_data = []
		self.stored_img_name = []
		self.image_data = []
		self.question_data = []
		self.answer_data = []
		self.voc_dict = None
		with open(dict_path, 'r') as f:
			self.voc_dict = json.load(f)['word2i']
		with open(json_path, 'r') as f:
			print('loading guesswhat json....')
			for line in tqdm(f):
				self.guesswhat_data.append(json.loads(line))
				break
		print('combine image and qa data....')
		for game in tqdm(self.guesswhat_data):
			image_name = game['image']['file_name']
			if image_name in self.stored_img_name:
				index = self.stored_img_name.index(image_name)
				img = self.image_data[index]
			else:
				img_type_dir = image_name.split('_')[1]
				img_path = os.path.join(*[img_dir, img_type_dir, image_name])
				img = imread(img_path)
			self.stored_img_name.append(image_name)
			qas = game['qas']
			for qa in qas:
				self.image_data.append(img)
				self.question_data.append(self.ques2vec(qa['question'][:-1]))
				if qa['answer'] == 'Yes':
					self.answer_data.append(1)
				else:
					self.answer_data.append(0)
			break
	def __len__(self):
		return len(self.image_data)
	def __getitem__(self, idx):
		return self.image_data[idx], self.question_data[idx], self.answer_data[idx]

	def ques2vec(self, question_str):
		vec_list = [self.voc_dict['<start>']]
		for voc in question_str.split(' '):
			voc_low = voc.lower()
			if voc_low not in self.voc_dict:
				vec_list.append(self.voc_dict['<unk>'])
			else:
				vec_list.append(self.voc_dict[voc_low])
		vec_list.append(self.voc_dict['<stop>'])
		return vec_list

def get_loader(img_dir, json_path, dict_path, batch_size, shuffle=True):
	dataset = GuessWhatDataset(img_dir, json_path, dict_path)
	data_loader = DataLoader(dataset=dataset,
				 batch_size=batch_size,
				 shuffle=shuffle,
				 drop_last=True)
	return data_loader
