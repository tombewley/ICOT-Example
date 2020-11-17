"""
Note that in this example, the OCT tree and ICOT (OCT warm-start, or no warm-start) result in the same solution. 
This is not generally the case, but can occur when the data is easily separated (as in the ruspini dataset)

The score printed in the browser view is negative to reflect the formulation as a minimization problem.
The positive score returned by the ICOT.score() functions reflect the "correct" interpretation of the score, 
in which we seek to maximize the criterion (with a maximum value of 1).

For larger datasets, we recommend setting warm_start = :oct and geom_threshold = 0.99 to improve the solve time.
"""

using DataFrames, MLDataUtils
using Clustering, Distances
using CSV
using Random
using Logging

# Set up Logging - we recommend to use this command to avoid package warnings during the model training process.
logger = Logging.SimpleLogger(stderr,Logging.Warn);
global_logger(logger);

##### Set parameters for the learners
"""Defines the internal validation criterion used to train the ICOT algorithm. 
Accepts two options :dunnindex (Dunn 1974) and :silhouette (Rousseeuw 1987)."""
criterion = :dunnindex
"""Provides a warmstart solution to initialize the algorithm. 
Details are provided in Section 3.3.2 of the paper. 
It can take as input :none, :greedy, and :oct. 
The OCT option uses user-selected labels (i.e. from K-means) 
to fit an Optimal Classification Tree as a supervised learning problem
to provide a warm-start to the algorithm. 
The greedy option fits a CART tree to these labels."""
warm_start = :none;
"""A boolean parameter that controls where the algorithm will enable
the geometric component of the feature space search. 
See details in Section 3.3.1 of the paper."""
geom_search = false
"""The percentile of gaps that will be considered 
by the geometric search for each feature. For example: 0.99."""
geom_threshold = 0.0
"""Integer specifying the number of random restarts to use in the local search algorithm. 
Must be positive and defaults to 100. The performance of the tree typically increases
as this value is increased, but with quickly diminishing returns. 
The computational cost of training increases linearly with this value."""
ls_num_tree_restarts = 100
"""Complexity parameter that determines the tradeoff between the accuracy and complexity
of the tree to control overfitting, as commonly seen in supervised learning problems.
The internal validation criteria used for this unsupervised algorithm naturally limit
the tree complexity, so we recommend to set the value to 0.0."""
cp = 0.0
"""The minimum number of points that must be present in every leaf node of the tree."""
minbucket = 10
"""The maximum depth of the fitted tree. This parameter must always be explicitly set or tuned. 
We recommend tuning this parameter using the grid search process described in the guide to parameter tuning."""
max_depth = 5
"""Integer controlling the randomized state of the algorithm. 
We recommend to set the seed to ensure reproducability of results."""
seed = 1

##### Step 1: Prepare the data
# Read the data - recommend the use of the (deprecated) readtable() command to avoid potential version conflicts with the CSV package.
data = readtable("../data/ruspini.csv"); 

# Convert the dataset to a matrix
data_array = convert(Matrix{Float64}, data);
# Get the number of observations and features
n, p = size(data_array)
data_t = data_array';

##### Step 2: Fit K-means clustering on the dataset to generate a warm-start for ICOT

# Fix the seed
Random.seed!(seed);

# The ruspini dataset has pre-defined clusters, which we will use to select the cluster count (K) for the K-means algorithm. 
# In an unsupervised setting (with no prior-known K), the number of clusters for K means can be selected using the elbow method.
K = length(unique(data_array[:,end]))

# Run k-means and save the assignments 
kmeans_result = kmeans(data_t, K);
assignment = kmeans_result.assignments;

data_full = DataFrame(hcat(data, assignment, makeunique=true));
names!(data_full, [:x1, :x2, :true_labels, :kmean_assign]);

##### Step 3: Run ICOT

# Prepare data for ICOT: features are stored in the matrix X, and the warm-start labels are stored in y
X = data_full[:,1:2]; y = data_full[:,:true_labels];

# Run ICOT with no warm-start: 
warm_start= :none
lnr_ws_none = ICOT.InterpretableCluster(
										ls_num_tree_restarts = ls_num_tree_restarts, 
										ls_random_seed = seed, 
										cp = cp, 
										max_depth = max_depth,
										minbucket = minbucket, 
										criterion = criterion, 
										ls_warmstart_criterion = criterion, 
										kmeans_warmstart = warm_start,
										geom_search = geom_search, 
										geom_threshold = geom_threshold
										);
run_time_icot_ls_none = @elapsed ICOT.fit!(lnr_ws_none, X, y);

ICOT.showinbrowser(lnr_ws_none)

# Run ICOT with an OCT warm-start: 
# Fit an OCT as a supervised learning problem with labels "y" and use this as the warm-start
warm_start= :oct
lnr_ws_oct = ICOT.InterpretableCluster(
									   ls_num_tree_restarts = ls_num_tree_restarts, 
									   ls_random_seed = seed, 
									   cp = cp, 
									   max_depth = max_depth,
									   minbucket = minbucket, 
									   criterion = criterion, 
									   ls_warmstart_criterion = criterion, 
									   kmeans_warmstart = warm_start,
									   geom_search = geom_search, 
									   geom_threshold = geom_threshold
									   );
run_time_icot_ls_oct = @elapsed ICOT.fit!(lnr_ws_oct, X, y);

# Evaluate models using cluster criteria.
score_ws_none = ICOT.score(lnr_ws_none, X, y, criterion=:dunnindex);
score_al_ws_none = ICOT.score(lnr_ws_none, X, y, criterion=:silhouette);
println("No warm start: runtime=", run_time_icot_ls_none, " dunn=", score_ws_none, " silhouette=", score_al_ws_none)
score_ws_oct = ICOT.score(lnr_ws_oct, X, y, criterion=:dunnindex);
score_al_ws_oct = ICOT.score(lnr_ws_oct, X, y, criterion=:silhouette);
println("OCT warm start: runtime=", run_time_icot_ls_oct, " dunn=", score_ws_oct, " silhouette=", score_al_ws_oct)