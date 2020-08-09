# Copyright 2018 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Download and preprocess WMT17 ende training and evaluation datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import tarfile
import urllib
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sentencepiece as spm
from fairseq.data import indexed_dataset
import torch

_TRAIN_DATA_SOURCES = [
    {
        "url": "http://data.statmt.org/wmt17/translation-task/"
               "training-parallel-nc-v12.tgz",
        "input": "news-commentary-v12.de-en.en",
        "target": "news-commentary-v12.de-en.de",
    },
    {
        "url": "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        "input": "commoncrawl.de-en.en",
        "target": "commoncrawl.de-en.de",
    },
    {
        "url": "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        "input": "europarl-v7.de-en.en",
        "target": "europarl-v7.de-en.de",
    },
]

_EVAL_DATA_SOURCES = [
    {
        "url": "http://data.statmt.org/wmt17/translation-task/dev.tgz",
        "input": "newstest2013.en",
        "target": "newstest2013.de",
    }
]

# Vocabulary constants
_TARGET_VOCAB_SIZE = 33708  # Number of subtokens in the vocabulary list.
_VOCAB_FILE = "vocab.ende.%d" % _TARGET_VOCAB_SIZE

# Strings to inclue in the generated files.
_PREFIX = "wmt32k"
_COMPILE_TAG = "compiled"
_ENCODE_TAG = "encoded"
_TRAIN_TAG = "train"
_EVAL_TAG = "dev"  # Following WMT and Tensor2Tensor conventions, in which the
                   # evaluation datasets are tagged as "dev" for development.

def find_file(path, filename, max_depth=5):
  """Returns full filepath if the file is in path or a subdirectory."""
  for root, dirs, files in os.walk(path):
    if filename in files:
      return os.path.join(root, filename)

    # Don't search past max_depth
    depth = root[len(path) + 1:].count(os.sep)
    if depth > max_depth:
      del dirs[:]  # Clear dirs
  return None


###############################################################################
# Download and extraction functions
###############################################################################
def get_raw_files(raw_dir, data_source):
  """Return raw files from source. Downloads/extracts if needed.

  Args:
    raw_dir: string directory to store raw files
    data_source: dictionary with
      {"url": url of compressed dataset containing input and target files
       "input": file with data in input language
       "target": file with data in target language}

  Returns:
    dictionary with
      {"inputs": list of files containing data in input language
       "targets": list of files containing corresponding data in target language
      }
  """
  raw_files = {
      "inputs": [],
      "targets": [],
  }  # keys
  for d in data_source:
    input_file, target_file = download_and_extract(
        raw_dir, d["url"], d["input"], d["target"])
    raw_files["inputs"].append(input_file)
    raw_files["targets"].append(target_file)
  return raw_files


def download_report_hook(count, block_size, total_size):
  """Report hook for download progress.

  Args:
    count: current block number
    block_size: block size
    total_size: total size
  """
  percent = int(count * block_size * 100 / total_size)
  print("\r%d%%" % percent + " completed", end="\r")


def download_from_url(path, url):
  """Download content from a url.

  Args:
    path: string directory where file will be downloaded
    url: string url

  Returns:
    Full path to downloaded file
  """
  filename = url.split("/")[-1]
  found_file = find_file(path, filename, max_depth=0)
  if found_file is None:
    filename = os.path.join(path, filename)
    print("Downloading from %s to %s." % (url, filename))
    inprogress_filepath = filename + ".incomplete"
    inprogress_filepath, _ = urllib.request.urlretrieve(
        url, inprogress_filepath, reporthook=download_report_hook)
    # Print newline to clear the carriage return from the download progress.
    print()
    os.rename(inprogress_filepath, filename)
    return filename
  else:
    print("Already downloaded: %s (at %s)." % (url, found_file))
    return found_file


def download_and_extract(path, url, input_filename, target_filename):
  """Extract files from downloaded compressed archive file.

  Args:
    path: string directory where the files will be downloaded
    url: url containing the compressed input and target files
    input_filename: name of file containing data in source language
    target_filename: name of file containing data in target language

  Returns:
    Full paths to extracted input and target files.

  Raises:
    OSError: if the the download/extraction fails.
  """
  # Check if extracted files already exist in path
  input_file = find_file(path, input_filename)
  target_file = find_file(path, target_filename)
  if input_file and target_file:
    print("Already downloaded and extracted %s." % url)
    return input_file, target_file

  # Download archive file if it doesn't already exist.
  compressed_file = download_from_url(path, url)

  # Extract compressed files
  print("Extracting %s." % compressed_file)
  with tarfile.open(compressed_file, "r:gz") as corpus_tar:
    corpus_tar.extractall(path)

  # Return filepaths of the requested files.
  input_file = find_file(path, input_filename)
  target_file = find_file(path, target_filename)

  if input_file and target_file:
    return input_file, target_file

  raise OSError("Download/extraction failed for url %s to path %s" %
                (url, path))


def txt_line_iterator(path):
  """Iterate through lines of file."""
  with open(path, mode='r', newline='\n') as f:
    for line in f:
      yield line.strip()


def compile_files(data_dir, raw_files, tag):
  """Compile raw files into a single file for each language.

  Args:
    raw_dir: Directory containing downloaded raw files.
    raw_files: Dict containing filenames of input and target data.
      {"inputs": list of files containing data in input language
       "targets": list of files containing corresponding data in target language
      }
    tag: String to append to the compiled filename.

  Returns:
    Full path of compiled input and target files.
  """
  print("Compiling files with tag %s." % tag)
  filename = "%s-%s-%s" % (_PREFIX, _COMPILE_TAG, tag)
  input_compiled_file = os.path.join(data_dir, filename + ".lang1")
  target_compiled_file = os.path.join(data_dir, filename + ".lang2")

  with open(input_compiled_file, mode="w", newline='\n') as input_writer:
    with open(target_compiled_file, mode="w", newline='\n') as target_writer:
      for i in range(len(raw_files["inputs"])):
        input_file = raw_files["inputs"][i]
        target_file = raw_files["targets"][i]

        print("Reading files %s and %s." % (input_file, target_file))
        write_file(input_writer, input_file)
        write_file(target_writer, target_file)
  return input_compiled_file, target_compiled_file


def write_file(writer, filename):
  """Write all of lines from file using the writer."""
  for line in txt_line_iterator(filename):
    writer.write(line)
    writer.write("\n")


###############################################################################
# Data preprocessing
###############################################################################
def encode_and_save_files_utf8(
    tokenizer, data_dir, raw_files, tag, total_shards):
  """Save data from files as encoded example pairs in UT8 format.

  Args:
    tokenizer: Subtokenizer object that will be used to encode the strings.
    data_dir: The directory in which to write the examples
    raw_files: A tuple of (input, target) data files. Each line in the input and
      the corresponding line in target file will be saved in encoded format (vocab integer ids).
    tag: String that will be added onto the file names.
    total_shards: Number of files to divide the data into.

  Returns:
    List of all files produced.
  """
  filepath = os.path.join(data_dir + '/utf8', '{}-{}-{}'.format(_PREFIX, _ENCODE_TAG, tag))
  src_fnl_name = filepath + '.src'
  dst_fnl_name = filepath + '.dst'

  if all_exist([src_fnl_name, dst_fnl_name]):
    print("Files with tag %s already exist." % tag)
    return src_fnl_name, dst_fnl_name

  print("Saving files with tag %s." % tag)
  input_file = raw_files[0]
  target_file = raw_files[1]

  # Write examples to each shard in round robin order.
  tmp_filepath = filepath + ".incomplete"
  src_tmp_name = tmp_filepath + '.src'
  dst_tmp_name = tmp_filepath + '.dst'
  with open(src_tmp_name, mode='w', newline='\n') as src_writer:
    with open(dst_tmp_name, mode='w', newline='\n') as dst_writer:
      counter = 0
      for counter, (input_line, target_line) in enumerate(zip(txt_line_iterator(input_file), txt_line_iterator(target_file))):
        if counter > 0 and counter % 100000 == 0:
          print("\tSaving case %d." % counter)

        src_writer.write(' '.join([str(idx) for
            idx in tokenizer.encode_as_ids(input_line.strip())  + [tokenizer.eos_id()]]))
        dst_writer.write(' '.join([str(idx) for
            idx in tokenizer.encode_as_ids(target_line.strip()) + [tokenizer.eos_id()]]))

        src_writer.write('\n')
        dst_writer.write('\n')


  os.rename(src_tmp_name, src_fnl_name)
  os.rename(dst_tmp_name, dst_fnl_name)

  print("Saved %d Examples" % counter)

  return src_fnl_name, dst_fnl_name


def all_exist(filepaths):
  """Returns true if all files in the list exist."""
  for fname in filepaths:
    if not os.path.exists(fname):
      return False
  return True


def make_dir(path):
  if not os.path.exists(path):
    print("Creating directory %s" % path)
    os.makedirs(path)

def make_binary_dataset(input_file, output_file):

    ds = indexed_dataset.IndexedDatasetBuilder(output_file +'.bin')

    for line in open(input_file, 'r'):
        tensor = torch.IntTensor([int(x) for x in line.split()])
        ds.add_item(tensor)

    ds.finalize(output_file + '.idx')

def main(unused_argv):
  """Obtain training and evaluation data for the Transformer model."""
  make_dir(FLAGS.raw_dir)
  make_dir(os.path.join(FLAGS.data_dir, 'utf8'))

  # Get paths of download/extracted training and evaluation files.
  print("Step 1/4: Downloading data from source")
  train_files = get_raw_files(FLAGS.raw_dir, _TRAIN_DATA_SOURCES)
  eval_files = get_raw_files(FLAGS.raw_dir, _EVAL_DATA_SOURCES)
  train_files_flat = train_files["inputs"] + train_files["targets"]

  # Create tokenizer based on the training files.
  print("Step 2/4: Creating tokenizer and building vocabulary")
  vocab_file = os.path.join(FLAGS.data_dir, _VOCAB_FILE)
  train_file_str=','.join(train_files_flat)
  #TODO: for some reason python API for sentencepiece crashes in this container!
  import subprocess
  subprocess.run('spm_train --input={} --model_prefix={} --vocab_size={} --max_sentence_length={} --model_type={} --bos_id=-1 --pad_id=0 --eos_id=1 --unk_id=2'.format(train_file_str, vocab_file, _TARGET_VOCAB_SIZE, 327680, 'bpe').split())
  tokenizer = spm.SentencePieceProcessor()
  tokenizer.load('{}.model'.format(vocab_file))

  print("Step 3/4: Compiling training and evaluation data")
  compiled_train_files = compile_files(FLAGS.data_dir, train_files, _TRAIN_TAG)
  compiled_eval_files = compile_files(FLAGS.data_dir, eval_files, _EVAL_TAG)

  # Tokenize and save data as Examples in the TFRecord format.
  print("Step 4/4: Preprocessing and saving data")
  train_utf8_files = encode_and_save_files_utf8(
      tokenizer, FLAGS.data_dir, compiled_train_files, _TRAIN_TAG, 1)
  eval_utf8_files = encode_and_save_files_utf8(
      tokenizer, FLAGS.data_dir, compiled_eval_files, _EVAL_TAG, 1)

  #TODO: clean this:
  for f in train_utf8_files:
    if 'src' in f:
        make_binary_dataset(f, os.path.join(FLAGS.data_dir, 'train.en-de.en'))
    elif 'dst' in f:
        make_binary_dataset(f, os.path.join(FLAGS.data_dir, 'train.en-de.de'))
  for f in eval_utf8_files:
    if 'src' in f:
        make_binary_dataset(f, os.path.join(FLAGS.data_dir, 'valid.en-de.en'))
    elif 'dst' in f:
        make_binary_dataset(f, os.path.join(FLAGS.data_dir, 'valid.en-de.de'))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data_dir", "-dd", type=str, default="/tmp/translate_ende",
      help="[default: %(default)s] Directory for where the "
           "translate_ende_wmt32k dataset is saved.",
      metavar="<DD>")
  parser.add_argument(
      "--raw_dir", "-rd", type=str, default="/tmp/translate_ende_raw",
      help="[default: %(default)s] Path where the raw data will be downloaded "
           "and extracted.",
      metavar="<RD>")
  parser.add_argument(
      "--search", action="store_true",
      help="If set, use binary search to find the vocabulary set with size"
           "closest to the target size (%d)." % _TARGET_VOCAB_SIZE)
  parser.add_argument('--init_vocab_file', type=str, default=None,
          help="Initialize tokenizer with predefined vocab file")

  FLAGS, unparsed = parser.parse_known_args()
  main(sys.argv)
