#! /bin/sh
touch generated_sentences
chmod u+r generated_sentences
echo > generated_sentences
sentences=100
echo Generating $sentences sentences.
echo
for i in $(seq 1 $sentences); do
  echo Generating sentence $i
  python rnn.py generate data rnn_folder 12 >> generated_sentences
  echo
  echo >> generated_sentences
done
echo $sentences sentences generated!
