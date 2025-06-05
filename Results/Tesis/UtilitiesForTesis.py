def decrease_rate(initial, final):
    """
    Calculate the decrease rate between two values.
    
    Parameters:
        initial (float): The initial value.
        final (float): The final value.
        
    Returns:
        float: The decrease rate.
    """
    return (initial - final) / initial * 100
def increase_rate(initial, final):
    """
    Calculate the increase rate between two values.
    
    Parameters:
        initial (float): The initial value.
        final (float): The final value.
        
    Returns:
        float: The increase rate.
    """
    return (final - initial) / initial * 100

def rate_changes(initial, final):
    if initial > final:
        return -decrease_rate(initial, final)
    else:
        return increase_rate(initial, final)
if __name__ == "__main__":
    # Example usage
    initial_value = 6.22
    final_value = 19
    ratio= rate_changes(initial_value, final_value)
    print(f"Rate change: {ratio:.2f}%")