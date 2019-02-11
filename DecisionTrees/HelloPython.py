
import math
from collections import Counter


def main():


    format_string = "{:>10.3f}||{:<10.3f}"
    val1 = math.pi
    val2 = math.e

    formatted = format_string.format(val1, val2)
    print(formatted)


main()

