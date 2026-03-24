1. Improved F1:
for hit/true positive follow this methods: flatten all ground trurh spans as a union of 3 second aligned windows. flatten all the predicted spans similarly. Then the intersection is the hit rate, and you can also think of false positive and false negative from here.

2. Aux reward: Keep as it is. Reduce the weight to 0.15

3. Reward for correct prediction: this reward is one for exact match with ground truth otherwise 0. It should be weighted by 0.2


Other points: Please log these 3 rewards separately on wandb as well.
