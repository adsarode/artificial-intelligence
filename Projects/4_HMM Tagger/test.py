a = {"Ashish": {"Maths": 100, "Python":10}, "Harish": {"Maths": 10, "Python":1}, "Aarya": {"Maths": 100, "Python":100}}

# for key, val in a.items():
	# print((key, val))
	

# for key, val in a.items():
	# for k, v in val.items():
		# print(k,v)


a = {"Jack": {"Maths": 100, "Python":10}, "Joe": {"Maths": 10, "Python":1}, "Soleman": {"Maths": 99, "C++":100}}
for key, val in a.items():
	print((key, max((v, k) for k,v in val.items())[1]))