import numpy as np
from sympy import Matrix, pretty, init_printing

# Enable pretty printing
init_printing()

def input_positive_int(prompt):
    """Prompt user for a positive integer, keep asking until valid."""
    while True:
        value = input(prompt).strip()
        if not value.isdigit() or int(value) == 0:
            print("⚠️  Please enter a positive integer greater than 0.")
        else:
            return int(value)

def input_matrix_row(expected_cols, row_num):
    """Prompt user for a row of numbers matching expected column count."""
    while True:
        row_input = input(f"Enter row {row_num} elements separated by space: ").split()
        if len(row_input) != expected_cols:
            print(f"⚠️  Row must have exactly {expected_cols} elements.")
            continue
        try:
            row = [float(x) if '.' in x else int(x) for x in row_input]
            return row
        except ValueError:
            print("⚠️  Please enter valid numbers (integers or decimals).")

def get_matrix_input(matrix_number):
    print(f"\nEnter elements for matrix {matrix_number}:")
    rows = input_positive_int("Number of rows: ")
    cols = input_positive_int("Number of columns: ")
    
    matrix = []
    for i in range(1, rows+1):
        row = input_matrix_row(cols, i)
        matrix.append(row)
    
    mat_np = np.array(matrix)
    print(f"\nMatrix {matrix_number} you entered:\n{mat_np}")
    return mat_np

def display_matrices_side_by_side(matrices):
    """Display matrices side by side for comparison."""
    rows = max(mat.shape[0] for mat in matrices)
    lines_per_matrix = [ [" ".join(map(str, row)) for row in mat] + [""]*(rows - mat.shape[0]) for mat in matrices ]
    
    for row_idx in range(rows):
        line = "   |   ".join(lines[row_idx].ljust(10) for lines in lines_per_matrix)
        print(line)

def main():
    print("Matrix Multiplication Program\n")
    
    num_matrices = input_positive_int("Enter the number of matrices to multiply: ")
    print(f"You will enter {num_matrices} matrices.\n")
    
    matrices = []
    for i in range(1, num_matrices+1):
        mat = get_matrix_input(i)
        matrices.append(mat)
    
    print("\nThe matrices that will be multiplied are (side by side):")
    display_matrices_side_by_side(matrices)
    
    while True:
        check = input("\nAre all matrices correct? (yes/no): ").strip().lower()
        if check in ['yes', 'no']:
            break
        print("⚠️  Please answer 'yes' or 'no'.")
    
    if check != 'yes':
        print("Please restart the program and enter the correct matrices.")
        return
    
    # Multiply matrices
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.dot(result, mat)
    
    # Display nicely using sympy
    sym_result = Matrix(result)
    print("\nResult of the multiplication:")
    print(pretty(sym_result))
    
    # LaTeX representation
    from sympy import latex
    print("\nLaTeX representation of the result:")
    print(latex(sym_result))

if __name__ == "__main__":
    main()