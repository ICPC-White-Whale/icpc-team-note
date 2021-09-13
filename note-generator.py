import os

codes = {}
for filename in os.listdir("algorithms"):
    path = os.path.join("algorithms", filename)
    stream = open(path, "r")
    codes[filename] = stream.read()
    stream.close()

md = "# 2021 ACM-ICPC Seoul Regional Team Note\n\n* **Team Name**: White Whale\n\n"
for name in codes:
    md += "## " + os.path.splitext(os.path.basename(name))[0] + "\n\n"
    md += "```c++\n"
    md += codes[name]
    md += "\n```\n\n"

stream = open("note.md", "w")
stream.write(md)
stream.close()