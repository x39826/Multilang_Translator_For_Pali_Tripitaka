import requests, json

# The format of the post data is "lang1|t_lang2 ......"  , 
# where lang1 is the name of source language, and lang2 is the target language translate to.

data = json.dumps([{'id':100, 'src':'tibt|t_romn པཡོགེཏིཔཡོཛ ེཏྭཱཨུཔཀྐམ ིཏུཾཨངྒཛཱཏཱམསན ཾ,པརསྶཨཱཎཱཔནནྟིཨེཝརཱུཔེཀཔཡོག ེ།'}])

r = requests.post("http://0.0.0.0:5000/translator/translate", data = data)
print(r.json())


r = requests.post("http://0.0.0.0:5000/translator/translate", data = json.dumps([{'id':100, 'src':'thai|t_thai รตฺติกํ ปุปฺผิตํ ทิสฺวาติ ปทุมปุปฺผาทีนิ อเนกานิ ปุปฺผานิ สูริยรํสิสมฺผเสฺสน ทิวา ปุปฺผนฺติ รตฺติยํ มกุฬิตานิ โหนฺติฯ'}]))
print(r.json())
