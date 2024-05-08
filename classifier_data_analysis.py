import json
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels

def main() -> int:
    """
    Run data analysis (paired t-tests) on the existing classifier results to see if there are significant differences in toxicity
    between multiple runs
    """

    outfile_path: str = "classifier_stats.json"

    data_paths: list[str] = [
            "/home/yz946/project/homework/cpsc_477_final_project/uncensored-llama-harmful-discussion-round-0-write.json",
            "/home/yz946/project/homework/cpsc_477_final_project/uncensored-llama-harmful-discussion-round-1-write.json",
            "/home/yz946/project/homework/cpsc_477_final_project/uncensored-llama-harmless-discussion-round-0-write.json",
            "/home/yz946/project/homework/cpsc_477_final_project/uncensored-llama-neutral-discussion-round-1-write.json",
            "/home/yz946/project/homework/cpsc_477_final_project/uncensored-llama-neutral-discussion-round-2-write.json",
            "/home/yz946/project/homework/cpsc_477_final_project/uncensored-llama-harmless-discussion-round-1-write.json",
            "/home/yz946/project/homework/cpsc_477_final_project/uncensored-llama-harmless-discussion-round-2-write.json",
 
            # "/home/yz946/project/homework/cpsc_477_final_project/uncensored-llama-harmful-discussion-round-2-write.json",
            # "/home/yz946/project/homework/cpsc_477_final_project/uncensored-llama-neutral-discussion-round-0-write.json",
    ]

    distributions: list[list[float]] = []
    for path in data_paths:
        lst: list[float] = get_list(path)
        # compute_stats(path)

        distributions.append(lst)

    # Do some analysis
    # Note: this function has some bugs
    # generate_stats(distributions, data_paths)

    # Create a dict to store the data and intention modes
    data_dict: dict[str: tuple[float, float]] = dict()
    for i, lst_1 in enumerate(distributions):
        for j, lst_2 in enumerate(distributions):
            if i != j:
                t_statistic, p_value = stats.ttest_rel(lst_1, lst_2)

                path_1_info: str = get_info(data_paths[i])
                path_2_info: str = get_info(data_paths[j])
                data_info: str = path_1_info + "_" +  path_2_info

                data: tuple[float, float] = (t_statistic, p_value)

                data_dict[data_info] = data


    # Save the data to a dict
    with open(outfile_path, "w") as outfile:
        json.dump(data_dict, outfile, indent=4)

    print(f"Data saved to {outfile_path}.")


    return 0

def generate_stats(distributions: list[list[float]], all_paths: list[str]) -> None:

    all_data: list[float] = []
    labels: list[str] = []
    for i, dist in enumerate(distributions):
        all_data.extend(dist)
        labels.extend([all_paths[i]] * len(dist))

    # Tukey's HSD Test
    tukey = statsmodels.stats.multicomp.pairwise_tukeyhsd(endog=all_data, groups=labels, alpha=0.05)
    tukey_summary = tukey.summary().data
    tukey_dict = [dict(zip(tukey_summary[0], row)) for row in tukey_summary[1:]]
    p_adj_values = [row[4] for row in tukey_summary[1:]]

    # Combine all results
    summary = p_adj_values

    # Extract the adjusted p-values

    # Save to JSON
    with open("anova_summary.json", "w") as file:
        print(summary)
        json.dump(summary, file, indent=4)

    print("ANOVA summary saved to 'anova_summary.json'.")


def get_info(path: str) -> str:
    if "llama" in path:
        model_name = "llama"
    else:
        model_name = "gpt"
    
    # Get intention
    data_intention: str = ""
    for intention in ["harmless", "neutral", "harmful"]:
        if intention in path:
            data_intention = intention
    
    # Get round number
    round_number: str = ""
    for round_nums in ["round-0", "round-1", "round-2"]:
        if round_nums in path:
            round_number = round_nums

    # info: tuple[str] = (model_name, data_intention, round_number)
    info: str = model_name + "-" + data_intention + "-" + round_number

    return info




def get_list(path: str) -> tuple[float]:
    with open(path, "r") as infile:

        file_data: dict[str: tuple[str, float]] = json.load(infile)

        # Extract the list of scores
        scores_list: list[float] = []
        for k in file_data.keys():
            new_score: float = file_data[k][1]
            scores_list.append(new_score)

    return scores_list


def compute_stats(path: str) -> tuple:
    with open(path, "r") as infile:

        file_data: dict[str: tuple[str, float]] = json.load(infile)

        # Extract the list of scores
        scores_list: list[float] = []
        for k in file_data.keys():
            new_score: float = file_data[k][1]
            scores_list.append(new_score)

        scores_list_np: np.array = np.array(scores_list)

        # Compute the mean and std of the scores list
        mean, std = scores_list_np.mean(), scores_list_np.std()
        print(f"Mean: {mean}, std: {std}")

        # Get quartiles
        scores_list_df: pd.DataFrame = pd.DataFrame(scores_list)
        quartiles: pd.DataFrame = scores_list_df.quantile([0, 0.25,0.5,0.75,1])
        print(f"Quartiles: \n{quartiles}")






if __name__ == "__main__":
    sys.exit(main())