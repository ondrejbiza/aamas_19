def edit_distance(key1, key2):
    """
    Edit distance between two keys.
    https://www.geeksforgeeks.org/edit-distance-dp-5/
    :param key1:        The first key.
    :param key2:        The second key.
    :returns:           Edit distance between the two keys.
    """

    m = len(key1)
    n = len(key2)

    memory = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):

            if i == 0:
                memory[i][j] = j
            elif j == 0:
                memory[i][j] = i
            elif key1[i - 1] == key2[j - 1]:
                memory[i][j] = memory[i - 1][j - 1]
            else:
                memory[i][j] = 1 + min(memory[i][j - 1], memory[i - 1][j], memory[i - 1][j - 1])

    return memory[m][n]
