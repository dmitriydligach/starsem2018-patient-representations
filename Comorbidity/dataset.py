#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')
import utils, i2b2
import numpy, pickle
import ConfigParser, os, nltk, pandas
import glob, string, collections, operator

# can be used to turn this into a binary task
LABEL2INT = {'Y':0, 'N':1, 'Q':2, 'U':3}
# file to log alphabet entries for debugging
ALPHABET_FILE = 'Model/alphabet.txt'

class DatasetProvider:
  """Comorboditiy data loader"""

  def __init__(self,
               corpus_path,
               annot_xml,
               disease,
               judgement,
               use_pickled_alphabet=False,
               alphabet_pickle=None,
               min_token_freq=0):
    """Index words by frequency in a file"""

    self.corpus_path = corpus_path
    self.annot_xml = annot_xml
    self.min_token_freq = min_token_freq
    self.disease = disease
    self.judgement = judgement
    self.alphabet_pickle = alphabet_pickle

    self.token2int = {}

    # when training, make alphabet and pickle it
    # when testing, load it from pickle
    if use_pickled_alphabet:
      print 'reading alphabet from', alphabet_pickle
      pkl = open(alphabet_pickle, 'rb')
      self.token2int = pickle.load(pkl)
    else:
      self.make_token_alphabet()

  def make_token_alphabet(self):
    """Map tokens (CUIs) to integers"""

    # count tokens in the entire corpus
    token_counts = collections.Counter()

    for f in os.listdir(self.corpus_path):
      file_path = os.path.join(self.corpus_path, f)
      file_feat_list = utils.read_cuis(file_path)
      token_counts.update(file_feat_list)

    # now make alphabet (high freq tokens first)
    index = 1
    self.token2int['oov_word'] = 0
    outfile = open(ALPHABET_FILE, 'w')
    for token, count in token_counts.most_common():
      if count > self.min_token_freq:
        outfile.write('%s|%s\n' % (token, count))
        self.token2int[token] = index
        index = index + 1

    # pickle alphabet
    pickle_file = open(self.alphabet_pickle, 'wb')
    pickle.dump(self.token2int, pickle_file)

  def load(self, maxlen=float('inf')):
    """Convert examples into lists of indices for keras"""

    labels = []    # int labels
    examples = []  # examples as int sequences
    no_labels = [] # docs with no labels

    # document id -> label mapping
    doc2label = i2b2.parse_standoff(
      self.annot_xml,
      self.disease,
      self.judgement)

    # load examples and labels
    for f in os.listdir(self.corpus_path):
      doc_id = f.split('.')[0]
      file_path = os.path.join(self.corpus_path, f)
      file_feat_list = utils.read_cuis(file_path)

      example = []
      # TODO: use unique tokens or not?
      for token in set(file_feat_list):
        if token in self.token2int:
          example.append(self.token2int[token])
        else:
          example.append(self.token2int['oov_word'])

      if len(example) > maxlen:
        example = example[0:maxlen]

      # no labels for some documents for some reason
      if doc_id in doc2label:
        string_label = doc2label[doc_id]
        int_label = LABEL2INT[string_label]
        labels.append(int_label)
        examples.append(example)
      else:
        no_labels.append(doc_id)

    print '%d documents with no labels for %s/%s in %s' \
      % (len(no_labels), self.disease,
         self.judgement, self.annot_xml.split('/')[-1])
    return examples, labels

  def load_vectorized(self, exclude, maxlen=float('inf')):
    """Same as above but labels are vectors"""

    labels = []    # int labels
    examples = []  # examples as int sequences
    no_labels = [] # docs with no labels

    # document id -> vector of labels
    doc2labels = i2b2.parse_standoff_vectorized(
      self.annot_xml,
      self.judgement,
      exclude)

    # load examples and labels
    for f in os.listdir(self.corpus_path):
      doc_id = f.split('.')[0]
      file_path = os.path.join(self.corpus_path, f)
      file_feat_list = utils.read_cuis(file_path)

      example = []
      # TODO: use unique tokens or not?
      for token in set(file_feat_list):
        if token in self.token2int:
          example.append(self.token2int[token])
        else:
          example.append(self.token2int['oov_word'])

      if len(example) > maxlen:
        example = example[0:maxlen]

      # no labels for some documents for some reason
      if doc_id in doc2labels:
        label_vector = doc2labels[doc_id]
        labels.append(label_vector)
        examples.append(example)
      else:
        no_labels.append(doc_id)

    print '%d documents with no labels for %s/%s in %s' \
      % (len(no_labels), self.disease,
         self.judgement, self.annot_xml.split('/')[-1])
    return examples, labels

  def load_raw(self):
    """Load for sklearn training"""

    labels = []    # string labels
    examples = []  # examples as strings
    no_labels = [] # docs with no labels

    # document id -> label mapping
    doc2label = i2b2.parse_standoff(
      self.annot_xml,
      self.disease,
      self.judgement)

    for f in os.listdir(self.corpus_path):
      doc_id = f.split('.')[0]
      file_path = os.path.join(self.corpus_path, f)
      file_feat_list = utils.read_cuis(file_path)

      # no labels for some documents for some reason
      if doc_id in doc2label:
        string_label = doc2label[doc_id]
        int_label = LABEL2INT[string_label]
        labels.append(int_label)
        examples.append(' '.join(file_feat_list))
      else:
        no_labels.append(doc_id)

    print '%d documents with no labels for %s/%s in %s' \
      % (len(no_labels), self.disease,
         self.judgement, self.annot_xml.split('/')[-1])
    return examples, labels

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, cfg.get('data', 'train_data'))
  annot_xml = os.path.join(base, cfg.get('data', 'train_annot'))

  dataset = DatasetProvider(data_dir, annot_xml)
  exclude = set(['GERD', 'Venous Insufficiency', 'CHF'])
  x, y = dataset.load_vectorized(exclude)
  print x
  print y
