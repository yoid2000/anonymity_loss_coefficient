def select_evenly_distributed_values(sorted_list):
    if len(sorted_list) <= 10:
        return sorted_list
    
    # Select the first and last values
    selected_values = [sorted_list[0]]
    
    # Calculate the step size for evenly distributed values
    step_size = (len(sorted_list) - 1) / 9
    
    # Select 8 evenly distributed values between the first and last values
    for i in range(1, 9):
        index = int(round(i * step_size))
        selected_values.append(sorted_list[index])
    
    # Add the last value
    selected_values.append(sorted_list[-1])
    
    return selected_values

# Example usage
sorted_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
# reverse the order of sorted_list
sorted_list = sorted_list[::-1]
selected_values = select_evenly_distributed_values(sorted_list)
print(sorted_list)
print(selected_values)