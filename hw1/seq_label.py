import numpy as np

#TODO:
# Note that, for the case when "B" and "C" are labeled with different values, you should label "A" with the minimum value of "B" and "C".

def getABCD(i,j, inpt):
	A = inpt[i][j]

	side_empty = 1

	if j == 0:
		B = side_empty
	else:
		B = inpt[i][j-1]

	if i == 0:
		C = side_empty
	else:
		C = inpt[i-1][j]

	if j == 0 or i == 0:
		D = side_empty
	else:
		D = inpt[i-1][j-1]

	return A, B, C, D

def seq_label(inpt, debug=True):
	orig = np.array(inpt, copy=True)

	count = 1
	for i in range(inpt.shape[0]):
		for j in range(inpt.shape[1]):
			A, B, C, D = getABCD(i, j, orig)
			nA, nB, nC, nD = getABCD(i, j, inpt)

			if debug:
				print("(i,j) = (", i, ", ", j, ")")
				print("ABCD")
				print(A,B,C,D)
				print("nABCD")
				print(nA, nB, nC, nD)
			if A == 0:
				if debug:
					print("c1")
				continue
			elif A == D:
				if debug:
					print("c2")
				inpt[i][j] = nD
			elif A != B:
				if debug:
					print("c3")
				if A == C:
					inpt[i][j] = nC
				else:
					inpt[i][j] = count
					count += 1
			elif A == C:
				if debug:
					print("c4")
				if B == C:
					inpt[i][j] = min(nB, nC)
				else:
					print("Update??? - impossible case")
			else:
				inpt[i][j] = nB
				if debug:
					print("c5")
	return inpt

inpt = [
	[0, 0, -1, -1, -1, -1, 0, -1],
	[0, -1, -1, -1, 0, -1, 0, -1],
	[-1, -1, -1, -1, 0, -1, 0, 0],
	[0, 0, 0, 0, 0, -1, 0, -1],
	[-1, -1, 0, 0, -1, -1, 0, -1],
	[-1, -1, -1, 0, 0, 0, 0, -1],
	[-1, 0, 0, 0, -1, 0, -1, -1],
	[-1, -1, -1, -1, -1, 0, -1, -1]
]

'''
[D, C]
[B, A]
'''

inpt = np.array(inpt)
print("raw")
print(inpt)

inpt = seq_label(inpt, debug=1)

print("first pass")
print(inpt)




