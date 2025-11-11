
def print_sorted(s):
	ret = ""
	s = s.upper()
	for c in "WELCOME TO MY KINGDOM":
		try:
			index = s.index(c)
			s = s[:index] + s[index + 1:]
			ret += c
		except:
			pass
	ret += s

	print(ret)

print_sorted("two milkmen go comedy")

while 1:
	line = input("> ")
	print_sorted(line)
