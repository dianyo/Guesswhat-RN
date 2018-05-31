import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvModel(nn.Module):
	def __init__(self):
		super(ConvModel, self).__init__()
			
		self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
		self.batchNorm1 = nn.BatchNorm2d(24)
		self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
		self.batchNorm2 = nn.BatchNorm2d(24)
		self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
		self.batchNorm3 = nn.BatchNorm2d(24)
		self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
		self.batchNorm4 = nn.BatchNorm2d(24)

	def forward(self, img):
		x = self.conv1(img)
		x = F.leaky_relu(x)
		x = self.batchNorm1(x)
		x = self.conv2(x)
		x = F.leaky_relu(x)
		x = self.batchNorm2(x)
		x = self.conv3(x)
		x = F.relu(x)
		x = self.batchNorm3(x)
		x = self.conv4(x)
		x = F.relu(x)
		x = self.batchNorm4(x)
		return x

class QuestionLSTM(nn.Module):
	def __init__(self, input_size, batch_size, seq_lengths, num_layers=2):
		super(QuestionLSTM, self).__init__()
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.lstm1 = nn.LSTM(input_size, 11, num_layers=num_layers, batch_first=True, bidirectional=True)
		self.fc1 = nn.Linear(128, 11)
		self.hidden = self.init_hidden()
		self.seq_lengths = seq_lengths
	def init_hidden(self):
		return (Variable(torch.zeros(2*self.num_layers, self.batch_size, 128)),
			Variable(torch.zeros(2*self.num_layers, self.batch_size, 128)))
	def forward(self, question):
		pack = nn.utils.rnn.pack_padded_sequence(question, self.seq_lengths, batch_first=True)
		out, self.hidden = self.lstm1(pack, self.hidden)
		x = self.fc1(out)
		return x

class RN(nn.Module):
	def __init__(self, batch_size, question_lengths):
		super(RN, self).__init__()
		
		self.conv = ConvModel()

		self.g_fc1 = nn.Linear((24+2)*2+11, 256)
		self.g_fc2 = nn.Linear(256, 256)
		self.g_fc3 = nn.Linear(256, 256)
		self.g_fc4 = nn.Linear(256, 256)

		self.f_fc1 = nn.Linear(256, 256)
		
		# move to cuda
		self.coord_oi = Variable(torch.FloatTensor(batch_size, 2))
		self.coord_oj = Variable(torch.FloatTensor(batch_size, 2))

		
		def cvt_coord(i):
			return	[(i/5-2)/2., (i%5-2)/2.]
		
		# move to cuda
		self.coord_tensor = Variable(torch.FloatTensor(batch_size, 25, 2))
		
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, 128)
		self.fc4 = nn.Linear(128, 2)

		self.questionLstm = QuestionLSTM(input_size=?, batch_size=batch_size, question_lengths=question_lengths)
	def forward(self, img, question):
		x = self.conv(img)

		mb = x.size()[0]
		n_channels = x.size()[1]
		d = x.size()[2]
		
		# x_flat = (64 x 25 x 24)
		x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
		
		# add coordinates
		x_flat = torch.cat([x_flat, self.coord_tensor],2)

		# add question everywhere
		qst = self.questionLstm(question)
		qst = torch.unsqueeze(qst, 1)
		qst = qst.repeat(1,25,1)
		qst = torch.unsqueeze(qst, 2)

		# cast all pairs against each other
		x_i = torch.unsqueeze(x_flat,1) # (64x1x25x26+11)
		x_i = x_i.repeat(1,25,1,1) # (64x25x25x26+11)
		x_j = torch.unsqueeze(x_flat,2) # (64x25x1x26+11)
		x_j = torch.cat([x_j,qst],3)
		x_j = x_j.repeat(1,1,25,1) # (64x25x25x26+11)

		# concatenate all together
		x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+11)

		# reshape for passing through network
		x_ = x_full.view(mb*d*d*d*d,63)
		x_ = self.g_fc1(x_)
		x_ = F.relu(x_)
		x_ = self.g_fc2(x_)
		x_ = F.relu(x_)
		x_ = self.g_fc3(x_)
		x_ = F.relu(x_)
		x_ = self.g_fc4(x_)
		x_ = F.relu(x_)

		# reshape again and sum
		x_g = x_.view(mb,d*d*d*d,256)
		x_g = x_g.sum(1).squeeze()

		"""f"""
		x_f = self.f_fc1(x_g)
		x_f = F.relu(x_f)

		x_out = self.fc2(x_f)
		x_out = F.leaky_relu(x_out)
		x_out = self.fc3(x_out)
		x_out = F.leaky_relu(x_out)
		x_out = self.fc4(x_out)
		return F.sigmoid(x)
