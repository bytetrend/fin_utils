import pandas as pd
import argparse

def merge_tsv_files(file1_path, file2_path, output_path):
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2, open(output_path, 'w') as out:
        # Read the first lines
        line1 = f1.readline()
        line2 = f2.readline()

        while line1 or line2:
            # Parse line 1 if available
            if line1:
                parts1 = line1.strip().split('\t')
                key1, val1 = parts1[0], float(parts1[1])
            else:
                key1 = None

            # Parse line 2 if available
            if line2:
                parts2 = line2.strip().split('\t')
                key2, val2 = parts2[0], float(parts2[1])
            else:
                key2 = None

            # Case 1: Both files still have data
            if key1 is not None and key2 is not None:
                if key1 == key2:
                    # Key matches: sum the numbers and advance both files
                    total = val1 + val2
                    # Convert to int if it's a whole number for cleaner output
                    total_str = str(int(total)) if total.is_integer() else str(total)
                    out.write(f"{key1}\t{total_str}\n")
                    line1 = f1.readline()
                    line2 = f2.readline()
                elif key1 < key2:
                    # Key 1 is smaller: write key 1 and advance file 1
                    val1_str = str(int(val1)) if val1.is_integer() else str(val1)
                    out.write(f"{key1}\t{val1_str}\n")
                    line1 = f1.readline()
                else:
                    # Key 2 is smaller: write key 2 and advance file 2
                    val2_str = str(int(val2)) if val2.is_integer() else str(val2)
                    out.write(f"{key2}\t{val2_str}\n")
                    line2 = f2.readline()

            # Case 2: Only file 1 has remaining data
            elif key1 is not None:
                val1_str = str(int(val1)) if val1.is_integer() else str(val1)
                out.write(f"{key1}\t{val1_str}\n")
                line1 = f1.readline()

            # Case 3: Only file 2 has remaining data
            elif key2 is not None:
                val2_str = str(int(val2)) if val2.is_integer() else str(val2)
                out.write(f"{key2}\t{val2_str}\n")
                line2 = f2.readline()

def parse_args():
    # Add arguments for input and output files
    parser = argparse.ArgumentParser(description='Merge trade counts from two files')
    parser.add_argument('-i1','--input1', help='Path to first input file')
    parser.add_argument("-i2","--input2", help='Path to second input file')
    parser.add_argument('-o','--output', help='Path to output file')
    return parser.parse_args()

def main():
    args = parse_args()
    # ... rest of the code that uses args.input1, args.input2, and args.output ...
    merge_tsv_files(args.input1, args.input2, args.output)

# Example usage:
# Replace 'file1.tsv' and 'file2.tsv' with your actual filenames
if __name__ == '__main__':
    main()
print("Files merged successfully!")
