def factor_x(slowest, fastest):
    return fastest / slowest
if __name__ == "__main__":
    # Example usage
    slowest = 4.25
    fastest = 13.4
    factor = factor_x(slowest, fastest)
    print(f"Factor: {factor:.2f}x")