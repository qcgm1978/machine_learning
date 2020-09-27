from math import log10, floor
from scipy.constants import codata

def most_significant_digit(x):
    e = floor(log10(x))
    return int(x*10**-e)

# count how many constants have each leading digit
constantsFren = [0]*10
d = codata.physical_constants
for c in d:
    (value, unit, uncertainty) = d[ c ]
    x = abs(value)
    constantsFren[most_significant_digit(x)] += 1
constantsFren=constantsFren[1:]
total = sum(constantsFren)


