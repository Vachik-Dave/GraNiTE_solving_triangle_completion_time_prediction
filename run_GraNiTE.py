import sys;
import argparse;
import math;
import pdb;
import numpy as np;
import networkx as nx;
from sklearn.metrics import mean_absolute_error;
from sklearn.metrics import mean_squared_error;
from sklearn.svm import LinearSVR;
#from sklearn import svm;
#from sklearn.metrics import classification_report;
import datetime
import graphlet_emb_triples_tctp
import node_emb_tctp


#Interval time thresholds
# all datasets
threshold1 = 30;
threshold2 = 60;

# DBLP dataset
#threshold1 = 2;
#threshold2 = 7;

#decay constant
lambd = 0.1;


def load_graph(filename):
	"""
	read graph from file
	"""
	f = open(filename);
	G =nx.Graph();
	index = 0;
	N = M = T = 0;
	time2edge = {};
	for line in f:
		index += 1;
		l = list(map(int,line.strip().split()) );
		if index == 1:
			N = l[0];
			M = l[1];
			T = l[2];
			continue;
		u = l[0];
		v = l[1];
		t = l[2];
		if G.has_edge(u,v):
			print("Repeated Edge: "+str(u)+"-"+str(v) );
			continue;

		G.add_edge(u,v);
		G[u][v]['time'] = t;

	return G,T;


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
		'''
		max_x = max(x);
		new_x = x;
		if max_x != 0:
			new_x = [each/max_x for each in x];
		'''
		new_x = [np.log(each+1) for each in x];

		Y.append(y);
		X.append(new_x);
		uv_list.append([u,v]);

	if train_start > 0 and count != len(Y):
		print("size mis-match for test!!!");

	return uv_list,X,Y,dim;


def load_emb_data(filename1):
	"""
	read stored embedding vectors from file
	"""
	index = 0;
	emb_dict = {}
	f = open(filename1);
	N = 0;
	dim = 0;
	for line in f:
		l = map(float,line.strip().split(","))
		if index == 0:
			N = int(l[0]);
			dim = int(l[1]);
			index += 1;
			continue;
		u = int(l[0]);
		try:
			f = emb_dict[u];
			print("repeat embedding ID: ",u);
		except KeyError:
			f = l[1:]
			emb_dict[u] = f;

	emb = [];
	for i in range(N):
		try:
			f = emb_dict[i];
			emb.append(f);
		except KeyError:
			print("no embedding for ID: ",i);

	return emb;


def process_k_triangles(svr_res):
	"""
	Find fix a single time-stamp for k-traingle links.
	"""
	global prev_edge_times;
	global edge2instance;
	global lambd;
	new_pred = [-999.99]*len(svr_res);
	for edge in edge2instance:
		l = edge2instance[edge];
		pred_weights = [];
		pred_edge_times = [];
		max_prev_edge_time = 0;
		for idx in l:
			p = svr_res[idx];
			w = 0.0;
			try:
				w = math.exp(-1.0*float(p));
				if w < 0.0000000001:
					w = 0.0000000001;
			except OverflowError:
				w = 0.0000000001;				# reduce the weight to very small value if exp() throws error
			#pred_weights.append( 1.0/(float(p)+1.0) );
			pred_weights.append(w);
			pred_edge_times.append(prev_edge_times[idx]+p);
			if max_prev_edge_time < prev_edge_times[idx]:
				max_prev_edge_time = prev_edge_times[idx];
		avg_edge_time = np.average(pred_edge_times,weights=pred_weights);
		if max_prev_edge_time > avg_edge_time:
			avg_edge_time = max_prev_edge_time;
		for idx in l:
			new_pred[idx] = avg_edge_time - prev_edge_times[idx];

	if -999.99 in new_pred:
		raise ValueError("Not all indexes are covered.")

	return new_pred;


def emb_features(graphlet_emb,train_x):
	return np.matmul(train_x,graphlet_emb);

def node_emb_feat(node_emb,uv_list):
	feat_vec = [];
	for each_uv in uv_list:
		u_vec = np.array(node_emb[each_uv[0]]);
		v_vec = np.array(node_emb[each_uv[1]]);	
		dist = np.absolute(u_vec-v_vec);
		feat_vec.append(dist);
	return np.array(feat_vec);


def parse_args():
	parser_arg = argparse.ArgumentParser(description = "GraNiTE framework for triangle completion time prediction.")
	
	parser_arg.add_argument('filename', type = str, default = '', help = 'original graph filename')
	parser_arg.add_argument('train_graphlet_filename', type = str, default = '', help = 'Training graphlet freq filename')
	parser_arg.add_argument('test_graphlet_filename', type = str, default = '', help = 'Testing graphlet freq filename')

	parser_arg.add_argument('--node_emb', type=str, action='store')
	parser_arg.add_argument('--graphlet_emb', type=str, action='store')

	parser_arg.add_argument('-d','--hidden_dim', type = int, default = 50, help = 'number of dimension for both embedding')
	parser_arg.add_argument('-r','--regulation_rate', type = float, default = 0.00001, help = 'embedding matrix regularization parameter')
	parser_arg.add_argument('-l','--learning_rate', type = float, default = 0.1, help = 'learning rate during min-batch gradient descent')
	parser_arg.add_argument('-e','--epochs', type = int, default = 25, help = 'epochs')
#	parser_arg.add_argument('batch_size', type = int, default = 100, help = 'min-batch size')

	return parser_arg.parse_args()


def main(args):
	global threshold1;
	global threshold2;

	# check if time-preserving node embedding file is provided
	if args.node_emb:
		start_t = datetime.datetime.now()
		node_emb = load_emb_data(args.node_emb);
		end_t = datetime.datetime.now();
		print("Node embedding reading time: "+str(end_t - start_t));

	else:	# run time preserving node embedding
		node_emb = node_emb_tctp.main(args);

	# check if graphlet based time-ordering embedding file is provided
	if args.graphlet_emb:
		start_t = datetime.datetime.now()
		graphlet_emb = load_emb_data(args.graphlet_emb);
		end_t = datetime.datetime.now();
		print("Graphlet embedding reading time: "+str(end_t - start_t));

	else:	# run graphlet based time-ordering embedding
		graphlet_emb = graphlet_emb_triples_tctp.main(args);


	start_t = datetime.datetime.now()

	G,T = load_graph(sys.argv[1]);
	train_uv,train_x,train_y,dim1 = load_graphlet_data(sys.argv[2],G,T*0.5);
	test_uv,test_x,test_y,dim2 = load_graphlet_data(sys.argv[3],G,-1);

	end_t = datetime.datetime.now();
	print("Reading time: "+str(end_t - start_t));


	# calculate embedding features
	new_train_x = node_emb_feat(node_emb,train_uv);
	new_test_x = node_emb_feat(node_emb,test_uv);
	new_train_x1 = emb_features(graphlet_emb,train_x);
	new_test_x1 = emb_features(graphlet_emb,test_x);

	# merge both embedding features
	new_train_x = np.column_stack((new_train_x,new_train_x1));
	new_test_x = np.column_stack((new_test_x,new_test_x1));
	

	#-----------------------------------------------------------------------------
	# regression

	start_t = datetime.datetime.now()

	# divide the interval times into bins based on thresholds
	small_idx = [];
	large_idx = [];	
	for i in range(len(test_y)):
		if test_y[i] <= threshold1:
			small_idx.append(i);
		elif test_y[i] <= threshold2:
			large_idx.append(i);

	test_y = np.array(test_y);
	small_test_y = test_y[small_idx];
	large_test_y = test_y[large_idx];

	avg_small_mae = 0.0;
	avg_large_mae = 0.0;

	# run the SVR
	for i in range(5):
		small_pred_y = [];
		large_pred_y = [];
		svr = LinearSVR(C=1.0, epsilon=0.5)	
		print("Learning...");
		sys.stdout.flush();
		svr.fit(new_train_x,train_y);
		print("Predicting...");
		sys.stdout.flush();
		svr_res = svr.predict(new_test_x);

		new_svr_res = process_k_triangles(svr_res);

	
		#fname = "LinearSVR_results.csv";
		#save_results(fname,test_y, new_svr_res);


		print("SVR Results:");
		new_svr_res = np.array(new_svr_res);
		small_pred_y = new_svr_res[small_idx];
		large_pred_y = new_svr_res[large_idx];


		print("Small Interval: "+str(threshold1)+" days");
		small_mae = mean_absolute_error(small_test_y, small_pred_y);
		avg_small_mae += small_mae;
		print("MAE: "+str(small_mae) );
		#print("RMSE: "+str(math.sqrt(mean_squared_error(small_test_y, small_pred_y))) );

		print("Large Interval: "+str(threshold1)+" to "+str(threshold2)+ " days");
		large_mae = mean_absolute_error(large_test_y, large_pred_y);
		avg_large_mae += large_mae;
		print("MAE: "+str(large_mae) );
		#print("RMSE: "+str(math.sqrt(mean_squared_error(large_test_y, large_pred_y))) );


	end_t = datetime.datetime.now();
	print 
	print("----------------------------")
	print
	print("Small Interval: "+str(threshold1)+" days");
	avg_small_mae /= 5.0;
	print("Avg. MAE: "+str(avg_small_mae) );

	print("Large Interval: "+str(threshold1)+" to "+str(threshold2)+ " days");
	avg_large_mae /= 5.0;
	print("Avg. MAE: "+str(avg_large_mae) );
	
	print("Regression time: "+str(end_t - start_t) );


if __name__=="__main__":
	args = parse_args();
	main(args);
