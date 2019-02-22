class A:
	def __init__(self, val1, val2):
		self.a1 = val1
		self.a2 = val2

A1 = A (10,20)
A2 = A (1,200)
A3 = A (5,500)

testDict = {A1: 'A1', A2: 'A2', A3: 'A3'}

best = max(testDict.keys(), key = lambda item: item.a2)
print(testDict[best])

#for a in testDict:
#	print(a.a1)