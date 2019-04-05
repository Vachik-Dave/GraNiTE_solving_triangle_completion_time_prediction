"""

author: Vachik Dave

"""

import sys;
import networkx as nx;
import math;
import random;
import numpy as np;
import tensorflow as tf;
from collections import defaultdict;
import argparse;
import time;


edge2instance = {};
prev_edge_times = [];

def load_graphlet_data(filename,G,train_start):
	"""
	read graphlet frequency vectors from file
	"""
	global prev_edge_times;
	global edge2instance;
	f = open(filename);
	index = 0;
	X = [];
	Y = [];
	uv_list = [];
	dim = 0;
	count = 0;
	for line in f:
		if index == 0:
			dim = len(line.strip().split(",")) - 4;				# node1,node2,interval_time,event_indicator,features...
			index += 1;
			continue;
		line = line.strip().split(",");
		if line[3] == '0':							#if event_indicator is false ignore the instance
			continue;

		u = int(line[0]);
		v = int(line[1]);
		if v < u:
			tmp = u;
			u = v;
			v = tmp;


		y = int(line[2]);
		second_edge_time = 0;
		if train_start > 0:						# training case
			time = G[u][v]['time'];
			second_edge_time = time - y;
			if train_start > second_edge_time:
				continue;
		else:								# testing
			time = G[u][v]['time'];
			second_edge_time = time - y;

			if (u,v) in edge2instance:				# stores instance indexes(count is current index and also count)
				edge2instance[(u,v)].append(count);
			else:
				edge2instance[(u,v)] = [count];

			count += 1;

			prev_edge_times.append(second_edge_time);


		x = list(map(float,line[4:]));
		max_x = max(x);
		new_x = x;
		if max_x != 0:
			new_x = [each/max_x for each in x];
		Y.append(y);
		X.append(new_x);
		uv_list.append([u,v]);

	if train_start > 0 and count != len(Y):
		print("size mis-match for test!!!");

	return uv_list,X,Y,dim;


def load_data(filename):
	"""
	read graph from file
	"""
	G = nx.Graph()
	max_T = -1;
	max_id = -1;
	index = -1;
	with open(filename, 'r') as f:
		for line in f:
			index += 1;
			if index == 0:
				continue;
			linetuple = line.strip().split()
			u = int(linetuple[0]);
			v = int(linetuple[1]);
			t = int(linetuple[2]);
			G.add_edge(u,v)
			G[u][v]['time'] = t;
			max_id = max(u, max_id)
			max_id = max(v, max_id)
			max_T = max(t,max_T);
	#print(max_id);
	return max_id, max_T, G; 



def generate_train_batch(G, uv_list,train_x,train_y, batch_size):
	"""
	generate batch of edge pairs <(u,v),(x,y)> where y_uv >= y_xy
	"""
	idxs1 = np.random.choice(len(train_y),batch_size);
	idxs2 = np.random.choice(len(train_y),batch_size);
	batch_Ydiff = [];
	N = len(G);
	batch_X = [];
	for i in range(batch_size):
		tpl = [];
		if train_y[idxs1[i]] >= train_y[idxs2[i]]:
			batch_Ydiff.append(train_y[idxs1[i]] - train_y[idxs2[i]]);
			tpl.append(uv_list[idxs1[i]][0]);
			tpl.append(uv_list[idxs1[i]][1]);
			tpl.append(uv_list[idxs2[i]][0]);
			tpl.append(uv_list[idxs2[i]][1]);
		else:
			batch_Ydiff.append(train_y[idxs2[i]] - train_y[idxs1[i]]);
			tpl.append(uv_list[idxs2[i]][0]);
			tpl.append(uv_list[idxs2[i]][1]);
			tpl.append(uv_list[idxs1[i]][0]);
			tpl.append(uv_list[idxs1[i]][1]);

		batch_X.append(tpl);

	return np.array(batch_X),np.array(batch_Ydiff)

def tctp_tf(N, hidden_dim, regulation_rate, learning_rate):
	"""
	build computational graph for time-preserving margin loss.
	"""
	x1 = tf.placeholder(tf.int32, [None])
	x2 = tf.placeholder(tf.int32, [None])
	x3 = tf.placeholder(tf.int32, [None])
	x4 = tf.placeholder(tf.int32, [None])
	y_diff = tf.placeholder(tf.float64,[None])

	emb_mat = tf.Variable(tf.random_normal([N+1, hidden_dim], stddev = 0.1,dtype=tf.float64), name = "emb")

	x1_emb = tf.nn.embedding_lookup(emb_mat, x1)
	x2_emb = tf.nn.embedding_lookup(emb_mat, x2)
	x3_emb = tf.nn.embedding_lookup(emb_mat, x3)
	x4_emb = tf.nn.embedding_lookup(emb_mat, x4)

	l2_norm = tf.add_n([
		tf.reduce_sum(tf.multiply(x1_emb, x1_emb)),
		tf.reduce_sum(tf.multiply(x2_emb, x2_emb)),
		tf.reduce_sum(tf.multiply(x3_emb, x3_emb)),
		tf.reduce_sum(tf.multiply(x4_emb, x4_emb))
		])
	
	e1_dist = tf.norm(x1_emb-x2_emb,ord = 2, axis = 1);
	e2_dist = tf.norm(x3_emb-x4_emb,ord = 2, axis = 1);

	relative_diff = (e1_dist - e2_dist) #/ (y_diff+1.0);
	margin_multiplier = 0.01;					# scale factor
	margin_threshold = y_diff * margin_multiplier; 
	obj = tf.reduce_mean(tf.nn.relu(margin_threshold - relative_diff ));

	my_loss = regulation_rate * l2_norm + obj;

	train_op = tf.train.AdagradOptimizer(learning_rate).minimize(my_loss)

	return x1, x2, x3, x4, y_diff, obj, my_loss, train_op, emb_mat;


def run_node_emb(N, T, G, uv_list,train_x,train_y, emb_filename,hidden_dim, regulation_rate, learning_rate, epochs, batch_size = 100):
	"""
	running the tensorflow computational graph
	"""
	num_iteration = int( float(len(train_y)) / float(batch_size) );
	with tf.Session() as session:
		x1, x2, x3, x4, y_diff, obj, my_loss, train_op, emb_mat = \
		    tctp_tf(N, hidden_dim, regulation_rate, learning_rate)
		print('construct tensorflow computational graph: Done!');
		session.run(tf.global_variables_initializer())

		for epoch in range(epochs):
			_batch_loss = 0
			for k in range(0, num_iteration): 
				batch_X, batch_Y = generate_train_batch(G,uv_list,train_x,train_y,batch_size);
				_loss, _train_opt = session.run([my_loss, train_op], \
							  feed_dict={x1:batch_X[:,0],x2:batch_X[:,1], x3:batch_X[:,2],  \
								x4:batch_X[:,3], y_diff:batch_Y[:]})
				_batch_loss += _loss

			print("epoch: ", epoch);
			print("my_loss: ", _batch_loss / float(num_iteration));

		final_emb_mat = session.run(emb_mat)
		#emb_filename = "bpr_emb_"+str(hidden_dim)+"_"+str(regulation_rate)+"_"+str(learning_rate)+"_"+str(epochs)+".txt"
		save_embedding(emb_filename,final_emb_mat);

		return final_emb_mat;


def save_embedding(filename,emb):
	f = open(filename,'w');
	N,dim = np.array(emb).shape;
	f.write(str(N)+","+str(dim));
	index = 0;
	for row in emb:
		f.write("\n"+str(index));
		index += 1;
		for ele in row:
			f.write(","+str(ele));
	f.close();


def parse_args():
	"""
	parse the embedding model arguments
	"""
	parser_arg = argparse.ArgumentParser(description = "Time-preserving node embedding.")
	
	parser_arg.add_argument('filename', type = str, default = '', help = 'embedding graph filename')
	parser_arg.add_argument('train_graphlet_filename', type = str, default = '', help = 'Training graphlet freq filename')
	parser_arg.add_argument('hidden_dim', type = int, default = 20, help = 'number of dimension')
	parser_arg.add_argument('regulation_rate', type = float, default = 0.0001, help = 'matrix regularization parameter')
	parser_arg.add_argument('learning_rate', type = float, default = 0.01, help = 'learning rate during min-batch gradient descent')
	parser_arg.add_argument('epochs', type = int, default = 20, help = 'epochs')
#	parser_arg.add_argument('batch_size', type = int, default = 100, help = 'min-batch size')

	return parser_arg.parse_args()


def main(args):
	start_time = time.time();
	N, T, G = load_data(args.filename);
	print('reading graphs: Done!')
	print("time elapsed: {:.2f}s".format(time.time() - start_time));
	sys.stdout.flush();	
	train_start = int(T*0.1);
	uv_list,train_x,train_y,dim1 = load_graphlet_data(args.train_graphlet_filename,G,train_start);

	emb_filename = sys.argv[1].strip().rsplit("_",2)[0].split("/")[-1];
	emb_filename += "_NodeEmb_";
	emb_filename += str(args.hidden_dim);
	emb_filename += "D_";
	emb_filename += str(args.learning_rate);
	emb_filename += "_";
	emb_filename += str(args.regulation_rate);
	emb_filename += "_";
	emb_filename += str(args.epochs);
	emb_filename += ".txt";

	print("learning_rate: "+str(args.learning_rate));
	print("regulation_rate: "+str(args.regulation_rate));

	start_time = time.time();
	node_emb = run_node_emb(N, T, G, uv_list,train_x,train_y, emb_filename,args.hidden_dim, args.regulation_rate,args.learning_rate, args.epochs)
	print("time elapsed: {:.2f}s".format(time.time() - start_time));

	return node_emb;

if __name__ == '__main__':
	args = parse_args()
	main(args)
