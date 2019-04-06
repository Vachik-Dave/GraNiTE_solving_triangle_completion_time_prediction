# GraNiTE 

### *Graphlet and Node based Time-conserving Embedding to solve triangle completion time prediction problem*.
#
#
This is a python implementation of GraNiTE framework that solves the triangle completion time prediction problem.

#### Requirements: 
- NumPy (tested version 1.14.3)
- tensoflow (tested version 1.10.0) 
- networkx (tested version 2.1) 
- scikit-learn (tested version 0.20.3)
> pip install -r requirements.txt

#### Run:

> python run_GraNiTE.py graph-filename train-graphlet-freq-filename test-graphlet-freq-filename 

###### Command line Options:
- \- \-node_emb <node_embedding_filename> (str) 
- \- \-graphlet_emb <graphlet_embedding_filename> (str)
- \-d \[\- \-hidden_dim\] <embedding_dimensions> (int)
- \-r \[\- \-regulation_rate\] <regularization_multiplier> (float)
- \-l \[\- \-learning_rate\] <learni_rate> (float)
- \-e \[\- \-epochs\] <num_epochs> (int)



##### Input formats:
1st input: grpah-filename (str)
 - Format: First row specify "num_nodes num_edges num_time-stamps". 
 - From the 2nd row, each row specify an edge in space delimited format: "node_id1 node_id2 time-stamp".
 - Node_id need to be integer and node_id starts with 0.
 - Time-stamps also need to be integer and starts with 0 (0th day).

2nd/3rd inputs: graphlet-freq-filename (str)
  - Format: First row headers "node_id1,node_id2,interval-time,event,x0,x1,x2,x3,...,x45"
  - Each row contains train/test sample (edge), corresponding interval-time and graphlet frequencies.
  - Note that, [graphlets g12 and g19](https://github.com/Vachik-Dave/E-CLoG-Counting-Edge-Centric-Local-Graphlets/blob/master/graphlets_5.pdf) are not local graphlet, so corresponding frequency x12 and x19 is always 0. Hence, effective graphlet dimension is 44.

##### Outputs:
- The code outputs MAE for small intervals (<= 30 days) and large intervals (31-60 days) for 5 runs of SVR and lastly prints average MAE for both interval tiems.
- The code also creates two embedding files with <graph-filename> as a prefix. These emebdding files can be passed as optional arguments (\- \-node_emb  OR \- \-graphlet_emb) for future runs.
