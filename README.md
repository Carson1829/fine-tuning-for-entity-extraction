# Homework 2 Data Files

Here are all the data files you\'ll need for Homework 2.

## Annotation files

`{train,validation}.json`: these json files can be read using
`pandas` as follows:

``` python
import pandas as pd

df = pd.read_json("train.json")
```

They contain the following columns:

  Field    Description
  -------- --------------------------------------------------------------
  annoid   A unique identifier for the annotation
  fileid   Name of the file the annotation is from
  start    Character start index of the annotation
  end      Character end index of the annotation
  tag      The annotation type (i.e., output class)
  text     The text contained in \`fileid\` in the range \[start, end\]

## File contents

`file_contents.json`: This file contains the source text for
the annotations. The keys should match up with the `fileid`
field in the annotation files, and the values are long strings
containing the file contents. As a sanity check, you can make sure the
`text` field in any given annotation equivalent to the
contents of `fileid` indexed at `[start, end]`:

``` python

# Grab the first row from the training data
df = pd.read_json('train.json')
row = df.iloc[0]

# Read the file contents file
with open('file_contents.json', 'r') as f:
    file_contents = json.load(f)

# For a given row, pull out the following fields for convenience
start = row['start']
end = row['end']
text = row['text']
fileid = row['fileid']

# Then grab the file contents from file_contents.json
content = file_contents[fileid]

# Make sure the contents are the same as the text
assert content[start:end] == text
```

## Unannotated MMD files

`./unannotated_mmds/`: These files are snippets of MMDs which
you are supposed to run inference on and then do your error analysis.
They are simply text files so you can read them in as you normally
would.

!pip install -q -U transformers datasets peft accelerate