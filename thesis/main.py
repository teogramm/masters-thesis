from thesis.files.filtering import add_summary_information
from thesis.files.observation_matching import add_matches
from thesis.files.processed import add_results_per_datapoint, add_results_per_id
from thesis.files.trajectories import open_trajectories
from thesis.preprocessing import preprocess
from thesis.processing import save_all_results
from thesis.processing.interactions import add_constrained_type
from thesis.results.report.graphs import riddarhuskajen, riddarholmsbron_n, riddarholmsbron_s


def main():
    preprocess()
    save_all_results()
    
    # Add all the calculated information to the trajectories
    trajectories = open_trajectories().lazy()
    trajectories = add_matches(trajectories)
    trajectories = add_summary_information(trajectories)
    trajectories = add_results_per_datapoint(trajectories)
    trajectories = add_results_per_id(trajectories)
    trajectories = add_constrained_type(trajectories)
    
    # Save the graphs
    riddarhuskajen(trajectories)
    riddarholmsbron_n(trajectories)
    riddarholmsbron_s(trajectories)
    
if __name__ == "__main__":
    main()