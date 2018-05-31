import argparse
import sys
import os
import numpy as np
import torch
from torch.autograd import Variable
from model import RN
from data import get_loader
from tqdm import tqdm
def main(args):
	data_loader = get_loader(args.image_dir, args.guesswhat_jsonl, args.voc_dict, args.batch_size)
	model = RN(args.batch_size, ???)
	criterion = nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
	total_step = len(data_loader)
	total_loss = 0
	hidden = None
	for epoch in tqdm(range(args.epochs)):
		for i, (img, question, answer) in enumerate(data_loader):
			model.hidden = model.init_hidden()
			variable_img = Variable(img)
			variable_question = Variable(question)
			variable_answer = Variable(answer)
			optimizer.zero_grad()
			output = model(variable_img, variable_question)
			loss = criterion(outpus, loss)
			loss.backward()
			optimizer.step()
			
			if (i+1)%10 == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss:{:.2f}'.format(epoch+1, args.epochs, i+1, total_step, loss.data[0]))
			total_loss += loss.data[0]
		torch.save(model.state_dict(), 'ckpt/train_001/epoch{}_loss{:.4f}'.format(epoch, total_loss/len(data_loader)))
		total_loss = 0
if __name__ == '__main___':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_dir', type=str, help='directory with images')
	parser.add_argument('--guesswhat_jsonl', type=str, help='the guesswhat jsonl file')
	parser.add_argumetn('--voc_dict', type=str, help='voc2index dictionary')
	parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
	parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
	parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
	parser.add_argument('--ckpt_dir', type=str, help='checkpoint stored directory')
	parser.add_argument('--use_ckpy', type=bool, default=False, help='continue training or not')
	args = parser.parse_args()
	if args.image_dir is None or args.guesswhat_jsonl is None:
		print('input dir error')
		sys.exit(0)
	main(args)
