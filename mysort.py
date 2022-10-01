import sys

# Read command line arguments and convert to a list of integers
arr = sys.argv[1].split(',')
my_numbers = [None]*len(arr)
for idx, arr_val in enumerate(arr):
    my_numbers[idx] = int(arr_val)
# Print
print(f'Before sorting {my_numbers}')

# My sorting (e.g. bubble sort)
# ADD HERE YOUR CODE
length = len(my_numbers)
swap = False
for i in range(length):
    for i2 in range(1, length):
        if my_numbers[i2] < my_numbers[i2-1]:
            my_numbers[i2-1], my_numbers[i2] = my_numbers[i2], my_numbers[i2-1]
            swap = True
    if not swap:
        break
# Print
print(f'After sorting {my_numbers}')