# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# import platform
# print(platform.machine())

from nomic.gpt4all import GPT4All

m = GPT4All()
m.open()
out = m.prompt('write me a short poem')
#out = m.generate('what is llm')
print(out)