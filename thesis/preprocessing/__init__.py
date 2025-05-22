from thesis.preprocessing.summary import preprocess_summary
from thesis.preprocessing.trajectories import preprocess_trajectories
from thesis.preprocessing.observations import preprocess_observations


def preprocess():
    preprocess_trajectories()
    preprocess_summary()
    preprocess_observations()
