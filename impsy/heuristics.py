import numpy as np

def rhythmic_consistency(branch: np.ndarray) -> float:
    # Return the standard deviation of the first elements in each array within this 2d array
    return -np.std(branch[:, 0])

def four_note_repetition(branch: np.ndarray) -> float: #TODO make this work for more than 2D
    branches_mod_4 = [branch[np.arange(len(branch)) % 4 == i] for i in range(4)]
    std_total = 0
    # Subtract the standard deviation of each mod arrays time and pitch, add the overall time and pitch standard deviation for the last 4 elements
    for b in branches_mod_4:
        if b.size > 0:
            std_total -= np.std(b[:, 0])
            std_total -= np.std(b[:, 1])
    std_total += np.std(branch[-4:, 0])
    std_total += np.std(branch[-4:, 1])
    return std_total


