import sys

def read_matrix(filename):
	f=open(filename,'r')
	m=int(f.readline().split()[0])
	mat=[]
	for i in range(m):
		row=[int(i) for i in f.readline().split()]
		mat.append(row)
	f.close()
	return mat

def sum(mat):
	s=0
	for row in mat:
		for elem in row:
			s+=elem
	return s

def maximum(mat):
	m=mat[0][0]
	for row in mat:
		for elem in row:
			if elem>m:
				m=elem
	return m

def mean(mat):
	return sum(mat)/(len(mat)*len(mat[0]))

def median(mat):
	l=[]
	for row in mat:
		l.extend(row)
	l.sort()
	n=len(l)
	# consider median = middle element if total number of elements is odd
	# else median = mean of middle two elements
	return l[int(n/2)] if n%2==1 else (l[int(n/2)-1]+l[int(n/2)])/2

def frequency_distribution(mat):
	freq={}
	for row in mat:
		for elem in row:
			freq[elem]=freq.get(elem,0)+1
	return freq

def mode(mat):
	freq=frequency_distribution(mat)
	max_freq=freq[max(freq,key=freq.get)]
	mode=[]
	for i in freq:
		if freq[i]==max_freq:
			mode.append(i)
	return mode

def standard_deviation(mat):
	mat_with_sq_entires=[[elem**2 for elem in row] for row in mat]
	return mean(mat_with_sq_entires)-mean(mat)**2

def print_matrix(mat):
	x=len(str(maximum(mat)))
	for row in mat:
		print('    ',end='')
		for elem in row:
			print(f"{elem:^{x}}",end=' ')
		print()

def print_frequency(freq):
	x=len(str(max(freq.keys())))
	for i in sorted(freq.keys()):
		print(f"    {i:^{x}} {'█'*freq[i]} {freq[i]}")

def print_all_statistics(mat):
		print_matrix(mat)
		print(f"sum = {sum(mat)}")
		print(f"maximum = {maximum(mat)}")
		print(f"mean = {mean(mat)}")
		print(f"median = {median(mat)}")
		md=mode(mat)
		if len(md)==1:
			print(f"mode = {md[0]}")
		else:
			print(f"set of modes = {md}")
		print(f"standard deviation = {standard_deviation(mat)}")
		print(f"frequency distribution:")
		freq=frequency_distribution(mat)
		print_frequency(freq)

if __name__ == "__main__":

	n=len(sys.argv)
	if(n<2):
		print(f"correct usage: python3 {sys.argv[0]} matrix_file_1 [more_matrix_files]")
		sys.exit()

	for filename in sys.argv[1:]:
		print(filename+":")
		mat=read_matrix(filename)
		print_all_statistics(mat)
		print("="*40)