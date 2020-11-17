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
data = readtable("data/ruspini.csv"); 
# Convert the dataset to a DataFrame
data_full = DataFrame(data);
# Prepare data for ICOT: features are stored in the matrix X, and the warm-start labels are stored in y
X = data_full[:,1:2]; y = data_full[:,:3];

##### Step 2: Run ICOT

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

# function my_apply(lnr::ICOT.InterpretableCluster, X::DataFrame)
#     [my_apply(lnr, X, i) for i = 1:nrow(X)]
# end
# function my_apply(lnr::ICOT.InterpretableCluster, X::DataFrame, i::Int)
#     t = 1
#     while true
#         if ICOT.IAI.is_leaf(lnr, t)
#             return t
#         end

#         if ICOT.IAI.is_hyperplane_split(lnr, t)
#             numeric_weights, categoric_weights = ICOT.IAI.get_split_weights(lnr, t)

#             split_value = 0.0
#             value_missing = false

#             for (feature, weight) in numeric_weights
#                 x = X[i, feature]
#                 value_missing = value_missing | ismissing(x)
#                 split_value += weight * x
#             end
#             for (feature, level_weights) in categoric_weights
#                 x = X[i, feature]
#                 value_missing = value_missing | ismissing(x_$feature)
#                 for (level, weight) in level_weights
#                     if x == level
#                         split_value += weight
#                     end
#                 end
#             end

#             threshold = ICOT.IAI.get_split_threshold(lnr, t)
#             goes_lower = split_value < threshold

#         else
#             feature = ICOT.IAI.get_split_feature(lnr, t)
#             x = X[i, feature]
#             value_missing = ismissing(x)

#             if ICOT.IAI.is_ordinal_split(lnr, t) ||
#                  ICOT.IAI.is_categoric_split(lnr, t) ||
#                  ICOT.IAI.is_mixed_ordinal_split(lnr, t)
#                 categories = ICOT.IAI.get_split_categories(lnr, t)
#                 goes_lower = categories[x]

#             elseif ICOT.IAI.is_parallel_split(lnr, t)
#                 threshold = ICOT.IAI.get_split_threshold(lnr, t)
#                 goes_lower = x < threshold

#             elseif ICOT.IAI.is_mixed_parallel_split(lnr, t)
#                 threshold = ICOT.IAI.get_split_threshold(lnr, t)
#                 categories = ICOT.IAI.get_split_categories(lnr, t)
#                 goes_lower = isa(x, Real) ? x < threshold : categories[x]
#             end
#         end

#         if value_missing
#             goes_lower = ICOT.IAI.missing_goes_lower(lnr, t)
#         end

#         t = goes_lower ? get_lower_child(lnr, t) : get_upper_child(lnr, t)
#     end
# end

# println(my_apply(lnr_ws_none.tree_, X))

# T = ICOT.InterpretableCluster
# for (name, typ) in zip(fieldnames(T), T.types)
#     println("$name = $typ")
# end

# Can indirectly access tree attributes by writing to a dot file (pretty dubious...)
write("tree.dot", ICOT.tree_to_dot(lnr_ws_none))