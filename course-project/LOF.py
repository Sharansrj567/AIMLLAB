# In Local Outlier Factor (LOF), the idea revolves around the concept of local regions. 
# Here, we calculate and compare the local density of the focus point with the local density of its neighbours. 
# If we find that the local density of the focus point is very low compared to its neighbours, 
# that would kind of hint that the focus point is isolated in that space and is a potential outlier. 
# The algorithm depends on the hyperparameter K, which decides upon the number of neighbours to consider when calculating the local density. 
# This value is bounded between 0 (no neighbour) and the total points (all points being neighbour) in the space.
# The local density function is defined as the reciprocal of average reachability distance, where, 
# average reachability distance is defined as the average distance from the focus point to all points in the neighbour.
#LOF = average local density of neighbors / local density of focus point

# If,
# LOF ≈ 1 similar density as neighbors
# LOF < 1 higher density than neighbors (normal point)
# LOF > 1 lower density than neighbors (anomaly)

from sklearn.neighbors import LocalOutlierFactor
data = [[1, 1], [2, 2.1], [1, 2], [2, 1], [50, 35], [2, 1.5]]
lof = LocalOutlierFactor(n_neighbors=2, metric='manhattan')
prediction = lof.fit_predict(data)
print(prediction)