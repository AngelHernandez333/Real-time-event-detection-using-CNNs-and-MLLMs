def factor_x(slowest, fastest):
    return fastest / slowest
if __name__ == "__main__":
    # Example usage
    slowest = 6.22
    fastest = 18.99
    factor = factor_x(slowest, fastest)
    print(f"Factor: {factor:.2f}x")