import q1_matrix_statistics as q1
from random import randrange

r=int(input("enter number of rows: "))
c=int(input("enter number of columns: "))

mat=[[randrange(11) for i in range(c)] for i in range(r)]
print("random matrix:")
q1.print_all_statistics(mat)